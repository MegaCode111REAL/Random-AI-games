/*!
 * PhysicsJS - A from-scratch 2D physics engine
 * ---------------------------------------------
 * Features:
 *  - Vec2 math library
 *  - Rigid bodies: circles & convex polygons, with mass/inertia computed from shape
 *  - Per-particle (point-mass) physics via Verlet integration for soft bodies / cloth / ropes
 *  - Gravity, linear drag (air resistance), and spring forces
 *  - Distance constraints, spring constraints, pin (anchor) constraints
 *  - Soft bodies (mass-spring meshes) and cloth simulation
 *  - Broad-phase collision via spatial hash grid
 *  - Narrow-phase collision via SAT (Separating Axis Theorem) for polygons + circle special-cases
 *  - Sequential-impulse solver: restitution (bounce), Coulomb friction (static+kinetic), positional correction
 *  - Sleeping bodies for performance
 *  - Raycasting
 *  - Joints: distance joint, spring joint, revolute (pin) joint, motor joint
 *  - Fixed-timestep world stepping with sub-stepping & solver iterations
 *
 * Usage: include this file, then:
 *   const world = new Physics.World();
 *   const body = Physics.RigidBody.circle(x, y, radius, { restitution: 0.8 });
 *   world.addBody(body);
 *   function loop(dt){ world.step(dt); requestAnimationFrame(loop); }
 *
 * All physics is 2D. Units are arbitrary but consistent (treat as meters/kg/seconds
 * if you want "real" behavior; gravity default is 980 px/s^2 style for screen coords).
 */
(function (global) {
  'use strict';

  const EPS = 1e-9;

  // ======================================================================
  // Vec2 - 2D vector math
  // ======================================================================
  class Vec2 {
    constructor(x = 0, y = 0) { this.x = x; this.y = y; }

    clone() { return new Vec2(this.x, this.y); }
    set(x, y) { this.x = x; this.y = y; return this; }
    copy(v) { this.x = v.x; this.y = v.y; return this; }

    add(v) { return new Vec2(this.x + v.x, this.y + v.y); }
    sub(v) { return new Vec2(this.x - v.x, this.y - v.y); }
    mul(s) { return new Vec2(this.x * s, this.y * s); }
    div(s) { return new Vec2(this.x / s, this.y / s); }
    neg() { return new Vec2(-this.x, -this.y); }

    addEq(v) { this.x += v.x; this.y += v.y; return this; }
    subEq(v) { this.x -= v.x; this.y -= v.y; return this; }
    mulEq(s) { this.x *= s; this.y *= s; return this; }

    dot(v) { return this.x * v.x + this.y * v.y; }
    cross(v) { return this.x * v.y - this.y * v.x; } // scalar (z-component)
    crossScalar(s) { return new Vec2(-s * this.y, s * this.x); } // z-vec cross v

    lengthSq() { return this.x * this.x + this.y * this.y; }
    length() { return Math.sqrt(this.lengthSq()); }

    normalize() {
      const len = this.length();
      if (len < EPS) return new Vec2(0, 0);
      return new Vec2(this.x / len, this.y / len);
    }

    normal() { return new Vec2(-this.y, this.x); } // left-hand perpendicular
    rightNormal() { return new Vec2(this.y, -this.x) }

    rotate(angle) {
      const c = Math.cos(angle), s = Math.sin(angle);
      return new Vec2(this.x * c - this.y * s, this.x * s + this.y * c);
    }

    distanceTo(v) { return this.sub(v).length(); }
    distanceToSq(v) { return this.sub(v).lengthSq(); }

    lerp(v, t) { return new Vec2(this.x + (v.x - this.x) * t, this.y + (v.y - this.y) * t); }

    static zero() { return new Vec2(0, 0); }
    static fromAngle(a, len = 1) { return new Vec2(Math.cos(a) * len, Math.sin(a) * len); }
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  // ======================================================================
  // Particle - point mass, Verlet-integrated (great for soft bodies/cloth/ropes)
  // ======================================================================
  class Particle {
    constructor(x, y, opts = {}) {
      this.pos = new Vec2(x, y);
      this.prevPos = new Vec2(x - (opts.vx || 0) * (opts.dt0 || 0.016), y - (opts.vy || 0) * (opts.dt0 || 0.016));
      this.accel = new Vec2(0, 0);
      this.mass = opts.mass !== undefined ? opts.mass : 1;
      this.invMass = this.mass > 0 ? 1 / this.mass : 0;
      this.radius = opts.radius !== undefined ? opts.radius : 4;
      this.pinned = !!opts.pinned;
      this.damping = opts.damping !== undefined ? opts.damping : 0.005; // velocity loss per step (air drag)
      this.restitution = opts.restitution !== undefined ? opts.restitution : 0.5;
      this.friction = opts.friction !== undefined ? opts.friction : 0.2;
    }

    get velocity() {
      return this.pos.sub(this.prevPos);
    }
    set velocity(v) {
      this.prevPos = this.pos.sub(v);
    }

    applyForce(f) {
      // a = F/m
      if (this.invMass === 0) return;
      this.accel.addEq(f.mul(this.invMass));
    }

    // Verlet integration step
    integrate(dt) {
      if (this.pinned) { this.accel.set(0, 0); return; }
      const vel = this.pos.sub(this.prevPos).mul(1 - this.damping);
      const next = this.pos.add(vel).add(this.accel.mul(dt * dt));
      this.prevPos.copy(this.pos);
      this.pos.copy(next);
      this.accel.set(0, 0);
    }

    pin() { this.pinned = true; return this; }
    unpin() { this.pinned = false; return this; }
  }

  global.__PhysicsPartial = { Vec2, clamp, Particle, EPS };
})(typeof window !== 'undefined' ? window : globalThis);

// ==========================================================================
// Constraints operating on Particles (Verlet-style, position-based)
// ==========================================================================
(function (global) {
  'use strict';
  const { Vec2 } = global.__PhysicsPartial;

  class DistanceConstraint {
    constructor(a, b, opts = {}) {
      this.a = a; this.b = b;
      this.restLength = opts.length !== undefined ? opts.length : a.pos.distanceTo(b.pos);
      this.stiffness = opts.stiffness !== undefined ? opts.stiffness : 1; // 1 = rigid rod, <1 = springy
      this.tearDistance = opts.tearDistance || Infinity; // break the constraint if stretched past this
      this.broken = false;
    }
    solve() {
      if (this.broken) return;
      const a = this.a, b = this.b;
      const delta = b.pos.sub(a.pos);
      const dist = delta.length() || 0.0001;
      if (dist > this.tearDistance) { this.broken = true; return; }
      const diff = (dist - this.restLength) / dist;
      const invMassSum = a.invMass + b.invMass;
      if (invMassSum === 0) return;
      const correction = delta.mul(diff * this.stiffness / invMassSum);
      if (!a.pinned) a.pos.addEq(correction.mul(a.invMass));
      if (!b.pinned) b.pos.subEq(correction.mul(b.invMass));
    }
  }

  // Spring constraint: force-based (adds accel each frame) rather than position-based -
  // gives bouncier, more "real" spring (Hooke's law: F = -k*x - damping*v)
  class SpringConstraint {
    constructor(a, b, opts = {}) {
      this.a = a; this.b = b;
      this.restLength = opts.length !== undefined ? opts.length : a.pos.distanceTo(b.pos);
      this.k = opts.stiffness !== undefined ? opts.stiffness : 120; // spring constant
      this.damping = opts.damping !== undefined ? opts.damping : 6;
    }
    apply() {
      const a = this.a, b = this.b;
      const delta = b.pos.sub(a.pos);
      const dist = delta.length() || 0.0001;
      const dir = delta.div(dist);
      const stretch = dist - this.restLength;
      const relVel = b.velocity.sub(a.velocity).dot(dir);
      const forceMag = -this.k * stretch - this.damping * relVel;
      const force = dir.mul(-forceMag);
      a.applyForce(force.neg());
      b.applyForce(force);
    }
  }

  // Pin a particle to a fixed world point (rigid anchor, optionally with slack)
  class PinConstraint {
    constructor(particle, point) {
      this.particle = particle;
      this.point = point.clone();
    }
    solve() {
      if (this.particle.pinned) return;
      this.particle.pos.copy(this.point);
    }
  }

  // Angle constraint (keeps 3 particles near a target bend angle) - useful for stiff ropes
  class AngleConstraint {
    constructor(a, b, c, opts = {}) {
      this.a = a; this.b = b; this.c = c; // b is the joint/pivot
      this.stiffness = opts.stiffness !== undefined ? opts.stiffness : 0.5;
    }
    solve() {
      const { a, b, c } = this;
      const v1 = a.pos.sub(b.pos);
      const v2 = c.pos.sub(b.pos);
      const targetAngle = 0; // straighten toward collinear; customize as needed
      const currentAngle = Math.atan2(v1.cross(v2), v1.dot(v2));
      const diff = (currentAngle - targetAngle) * this.stiffness * 0.5;
      const cos = Math.cos(diff), sin = Math.sin(diff);
      const rotated = new Vec2(v2.x * cos - v2.y * sin, v2.x * sin + v2.y * cos);
      if (!c.pinned) c.pos.copy(b.pos.add(rotated));
    }
  }

  global.__PhysicsPartial.DistanceConstraint = DistanceConstraint;
  global.__PhysicsPartial.SpringConstraint = SpringConstraint;
  global.__PhysicsPartial.PinConstraint = PinConstraint;
  global.__PhysicsPartial.AngleConstraint = AngleConstraint;
})(typeof window !== 'undefined' ? window : globalThis);

// ==========================================================================
// Shapes: Circle & convex Polygon, with mass/inertia computation
// ==========================================================================
(function (global) {
  'use strict';
  const { Vec2 } = global.__PhysicsPartial;

  class CircleShape {
    constructor(radius) {
      this.type = 'circle';
      this.radius = radius;
    }
    computeMass(density) {
      const area = Math.PI * this.radius * this.radius;
      const mass = area * density;
      // moment of inertia of solid disk about center: (1/2) m r^2
      const inertia = 0.5 * mass * this.radius * this.radius;
      return { mass, inertia, area };
    }
    getAABB(pos) {
      return { minX: pos.x - this.radius, minY: pos.y - this.radius, maxX: pos.x + this.radius, maxY: pos.y + this.radius };
    }
  }

  // Convex polygon defined by local-space vertices (CCW winding)
  class PolygonShape {
    constructor(vertices) {
      this.type = 'polygon';
      this.vertices = vertices.map(v => new Vec2(v.x, v.y));
      this._recenter();
      this.computeNormals();
    }

    static box(w, h) {
      const hw = w / 2, hh = h / 2;
      return new PolygonShape([
        new Vec2(-hw, -hh), new Vec2(hw, -hh), new Vec2(hw, hh), new Vec2(-hw, hh)
      ]);
    }

    static regular(sides, radius) {
      const verts = [];
      for (let i = 0; i < sides; i++) {
        const a = (i / sides) * Math.PI * 2;
        verts.push(new Vec2(Math.cos(a) * radius, Math.sin(a) * radius));
      }
      return new PolygonShape(verts);
    }

    _recenter() {
      // shift vertices so centroid is local origin (needed for correct inertia/rotation)
      const c = this.centroid();
      this.vertices = this.vertices.map(v => v.sub(c));
    }

    centroid() {
      let cx = 0, cy = 0, area = 0;
      const v = this.vertices;
      for (let i = 0; i < v.length; i++) {
        const p1 = v[i], p2 = v[(i + 1) % v.length];
        const cross = p1.cross(p2);
        area += cross;
        cx += (p1.x + p2.x) * cross;
        cy += (p1.y + p2.y) * cross;
      }
      area *= 0.5;
      if (Math.abs(area) < 1e-9) return new Vec2(0, 0);
      cx /= (6 * area); cy /= (6 * area);
      return new Vec2(cx, cy);
    }

    computeNormals() {
      const v = this.vertices;
      this.normals = v.map((p, i) => {
        const next = v[(i + 1) % v.length];
        const edge = next.sub(p);
        return edge.rightNormal().normalize();
      });
    }

    computeMass(density) {
      // polygon area, centroid (already centered), and inertia via standard formula
      let area = 0, inertia = 0;
      const v = this.vertices;
      const k = 1 / 3;
      for (let i = 0; i < v.length; i++) {
        const p1 = v[i], p2 = v[(i + 1) % v.length];
        const cross = Math.abs(p1.cross(p2));
        const triArea = 0.5 * cross;
        area += triArea;
        const intx2 = p1.x * p1.x + p1.x * p2.x + p2.x * p2.x;
        const inty2 = p1.y * p1.y + p1.y * p2.y + p2.y * p2.y;
        inertia += (0.25 * k * cross) * (intx2 + inty2);
      }
      const mass = area * density;
      const inertiaOut = inertia * density;
      return { mass, inertia: inertiaOut, area };
    }

    getWorldVertices(pos, angle) {
      return this.vertices.map(v => v.rotate(angle).add(pos));
    }
    getWorldNormals(angle) {
      return this.normals.map(n => n.rotate(angle));
    }

    getAABB(pos, angle) {
      const wv = this.getWorldVertices(pos, angle);
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const p of wv) {
        minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
        minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
      }
      return { minX, minY, maxX, maxY };
    }
  }

  global.__PhysicsPartial.CircleShape = CircleShape;
  global.__PhysicsPartial.PolygonShape = PolygonShape;
})(typeof window !== 'undefined' ? window : globalThis);

// ==========================================================================
// RigidBody - full 2D rigid body with linear + angular dynamics
// ==========================================================================
(function (global) {
  'use strict';
  const { Vec2, CircleShape, PolygonShape } = global.__PhysicsPartial;

  let _idCounter = 1;

  class RigidBody {
    constructor(shape, opts = {}) {
      this.id = _idCounter++;
      this.shape = shape;
      this.pos = new Vec2(opts.x || 0, opts.y || 0);
      this.angle = opts.angle || 0;
      this.velocity = new Vec2(opts.vx || 0, opts.vy || 0);
      this.angularVelocity = opts.angularVelocity || 0;

      this.isStatic = !!opts.isStatic;
      this.density = opts.density !== undefined ? opts.density : 1;

      const md = shape.computeMass(this.density);
      this.mass = this.isStatic ? 0 : md.mass;
      this.invMass = (this.isStatic || this.mass === 0) ? 0 : 1 / this.mass;
      this.inertia = this.isStatic ? 0 : md.inertia;
      this.invInertia = (this.isStatic || this.inertia === 0) ? 0 : 1 / this.inertia;

      this.restitution = opts.restitution !== undefined ? opts.restitution : 0.3; // bounciness 0-1
      this.staticFriction = opts.staticFriction !== undefined ? opts.staticFriction : 0.5;
      this.dynamicFriction = opts.dynamicFriction !== undefined ? opts.dynamicFriction : 0.3;
      this.linearDamping = opts.linearDamping !== undefined ? opts.linearDamping : 0.01; // air resistance
      this.angularDamping = opts.angularDamping !== undefined ? opts.angularDamping : 0.01;
      this.gravityScale = opts.gravityScale !== undefined ? opts.gravityScale : 1;

      this.force = new Vec2(0, 0);
      this.torque = 0;

      this.isSensor = !!opts.isSensor; // detects collisions but doesn't resolve them
      this.collisionGroup = opts.collisionGroup || 0; // bodies in same nonzero group can be set to not collide
      this.categoryBits = opts.categoryBits !== undefined ? opts.categoryBits : 0x0001;
      this.maskBits = opts.maskBits !== undefined ? opts.maskBits : 0xFFFF;

      this.sleeping = false;
      this.sleepTimer = 0;
      this.allowSleep = opts.allowSleep !== undefined ? opts.allowSleep : true;

      this.userData = opts.userData || null;
    }

    static circle(x, y, radius, opts = {}) {
      return new RigidBody(new CircleShape(radius), Object.assign({ x, y }, opts));
    }
    static box(x, y, w, h, opts = {}) {
      return new RigidBody(PolygonShape.box(w, h), Object.assign({ x, y }, opts));
    }
    static polygon(x, y, vertices, opts = {}) {
      return new RigidBody(new PolygonShape(vertices), Object.assign({ x, y }, opts));
    }
    static regularPolygon(x, y, sides, radius, opts = {}) {
      return new RigidBody(PolygonShape.regular(sides, radius), Object.assign({ x, y }, opts));
    }

    applyForce(f, worldPoint) {
      if (this.invMass === 0) return;
      this.wake();
      this.force.addEq(f);
      if (worldPoint) {
        const r = worldPoint.sub(this.pos);
        this.torque += r.cross(f);
      }
    }

    applyImpulse(impulse, worldPoint) {
      if (this.invMass === 0) return;
      this.wake();
      this.velocity.addEq(impulse.mul(this.invMass));
      if (worldPoint) {
        const r = worldPoint.sub(this.pos);
        this.angularVelocity += this.invInertia * r.cross(impulse);
      }
    }

    applyTorque(t) { this.torque += t; }

    wake() { this.sleeping = false; this.sleepTimer = 0; }

    getWorldVertices() {
      if (this.shape.type !== 'polygon') return null;
      return this.shape.getWorldVertices(this.pos, this.angle);
    }
    getWorldNormals() {
      if (this.shape.type !== 'polygon') return null;
      return this.shape.getWorldNormals(this.angle);
    }
    getAABB() {
      return this.shape.getAABB(this.pos, this.angle);
    }

    // velocity of a specific point on the body (linear + rotational contribution)
    velocityAtPoint(worldPoint) {
      const r = worldPoint.sub(this.pos);
      // v_point = v + omega x r  =>  omega x r = omega * perp(r)
      return this.velocity.add(r.crossScalar(this.angularVelocity));
    }

    integrate(dt, gravity) {
      if (this.isStatic || this.sleeping) { this.force.set(0, 0); this.torque = 0; return; }

      // Semi-implicit (symplectic) Euler integration - stable for real-time sims
      const accel = this.force.mul(this.invMass).add(gravity.mul(this.gravityScale));
      this.velocity.addEq(accel.mul(dt));
      this.velocity.mulEq(1 / (1 + this.linearDamping * dt));

      const angAccel = this.torque * this.invInertia;
      this.angularVelocity += angAccel * dt;
      this.angularVelocity *= 1 / (1 + this.angularDamping * dt);

      this.pos.addEq(this.velocity.mul(dt));
      this.angle += this.angularVelocity * dt;

      this.force.set(0, 0);
      this.torque = 0;
    }

    kineticEnergy() {
      const lin = 0.5 * this.mass * this.velocity.lengthSq();
      const rot = 0.5 * this.inertia * this.angularVelocity * this.angularVelocity;
      return lin + rot;
    }
  }

  global.__PhysicsPartial.RigidBody = RigidBody;
})(typeof window !== 'undefined' ? window : globalThis);

// ==========================================================================
// Collision detection: AABB overlap, SAT (polygon-polygon), circle cases
// Produces a "manifold": { bodyA, bodyB, normal, penetration, contacts[] }
// ==========================================================================
(function (global) {
  'use strict';
  const { Vec2 } = global.__PhysicsPartial;

  function aabbOverlap(a, b) {
    return a.minX <= b.maxX && a.maxX >= b.minX && a.minY <= b.maxY && a.maxY >= b.minY;
  }

  function circleVsCircle(a, b) {
    const ra = a.shape.radius, rb = b.shape.radius;
    const delta = b.pos.sub(a.pos);
    const dist = delta.length();
    const radiiSum = ra + rb;
    if (dist >= radiiSum || dist < 1e-9 && false) {
      if (dist >= radiiSum) return null;
    }
    const normal = dist > 1e-9 ? delta.div(dist) : new Vec2(0, -1);
    const penetration = radiiSum - dist;
    const contactPoint = a.pos.add(normal.mul(ra));
    return { normal, penetration, contacts: [contactPoint] };
  }

  // Find minimum-penetration axis for polygon vs polygon (SAT support function)
  function findAxisLeastPenetration(vertsA, normsA, vertsB) {
    let bestDist = -Infinity, bestIdx = -1;
    for (let i = 0; i < normsA.length; i++) {
      const n = normsA[i];
      const v = vertsA[i];
      // support point on B in direction -n (most negative projection)
      let minProj = Infinity;
      for (const bv of vertsB) {
        const proj = n.dot(bv.sub(v));
        if (proj < minProj) minProj = proj;
      }
      if (minProj > bestDist) { bestDist = minProj; bestIdx = i; }
      if (minProj > 0) return { separation: minProj, index: i }; // early out: separating axis found
    }
    return { separation: bestDist, index: bestIdx };
  }

  function polygonVsPolygon(a, b) {
    const vertsA = a.getWorldVertices(), normsA = a.getWorldNormals();
    const vertsB = b.getWorldVertices(), normsB = b.getWorldNormals();

    const ra = findAxisLeastPenetration(vertsA, normsA, vertsB);
    if (ra.separation > 0) return null;
    const rb = findAxisLeastPenetration(vertsB, normsB, vertsA);
    if (rb.separation > 0) return null;

    let refPoly, refVerts, refNorms, refIdx, incVerts, flip;
    if (rb.separation > ra.separation + 1e-6) {
      refVerts = vertsB; refNorms = normsB; refIdx = rb.index; incVerts = vertsA; flip = true;
    } else {
      refVerts = vertsA; refNorms = normsA; refIdx = ra.index; incVerts = vertsB; flip = false;
    }
    const refNormal = refNorms[refIdx];

    // find incident edge on the other polygon (most anti-parallel normal)
    let incIdx = 0, minDot = Infinity;
    const incNorms = flip ? normsA : normsB;
    for (let i = 0; i < incNorms.length; i++) {
      const d = refNormal.dot(incNorms[i]);
      if (d < minDot) { minDot = d; incIdx = i; }
    }
    const incEdgeStart = incVerts[incIdx];
    const incEdgeEnd = incVerts[(incIdx + 1) % incVerts.length];

    // clip incident edge against reference edge side planes (Sutherland-Hodgman, simplified)
    const refStart = refVerts[refIdx];
    const refEnd = refVerts[(refIdx + 1) % refVerts.length];
    const refEdgeDir = refEnd.sub(refStart).normalize();

    let contacts = [incEdgeStart, incEdgeEnd];
    contacts = clipSegment(contacts, refEdgeDir, refStart, -refEdgeDir.dot(refStart));
    if (contacts.length < 2) return null;
    contacts = clipSegment(contacts, refEdgeDir.neg(), refEnd, refEdgeDir.dot(refEnd));
    if (contacts.length < 2) return null;

    // keep only points below the reference face (penetrating)
    const finalContacts = [];
    let penetration = 0;
    for (const p of contacts) {
      const sep = refNormal.dot(p.sub(refStart));
      if (sep <= 0) { finalContacts.push(p); penetration = Math.max(penetration, -sep); }
    }
    if (finalContacts.length === 0) return null;

    const normal = flip ? refNormal.neg() : refNormal;
    return { normal, penetration, contacts: finalContacts };
  }

  function clipSegment(points, normal, linePoint, offset) {
    const out = [];
    const d0 = normal.dot(points[0]) + offset;
    const d1 = normal.dot(points[1]) + offset;
    if (d0 <= 0) out.push(points[0]);
    if (d1 <= 0) out.push(points[1]);
    if (d0 * d1 < 0) {
      const t = d0 / (d0 - d1);
      out.push(points[0].lerp(points[1], t));
    }
    return out;
  }

  function closestPointOnPolygon(pos, polyBody) {
    // returns { point, normal, distance } closest edge to a circle center
    const verts = polyBody.getWorldVertices();
    const norms = polyBody.getWorldNormals();
    let bestDist = -Infinity, bestIdx = 0;
    for (let i = 0; i < norms.length; i++) {
      const d = norms[i].dot(pos.sub(verts[i]));
      if (d > bestDist) { bestDist = d; bestIdx = i; }
    }
    return { edgeIndex: bestIdx, separation: bestDist, verts, norms };
  }

  function circleVsPolygon(circleBody, polyBody) {
    const c = circleBody.pos, r = circleBody.shape.radius;
    const { edgeIndex, separation, verts, norms } = closestPointOnPolygon(c, polyBody);
    if (separation > r) return null; // definitely no collision

    const v1 = verts[edgeIndex];
    const v2 = verts[(edgeIndex + 1) % verts.length];

    let normal, penetration, contactPoint;
    if (separation < 0) {
      // circle center inside polygon
      normal = norms[edgeIndex];
      penetration = r - separation;
      contactPoint = c.sub(normal.mul(r));
    } else {
      // circle center outside; determine region (edge / vertex voronoi regions)
      const u1 = c.sub(v1).dot(v2.sub(v1));
      const u2 = c.sub(v2).dot(v1.sub(v2));
      if (u1 <= 0) {
        const dist = c.distanceTo(v1);
        if (dist > r) return null;
        normal = c.sub(v1).normalize();
        penetration = r - dist;
        contactPoint = v1;
      } else if (u2 <= 0) {
        const dist = c.distanceTo(v2);
        if (dist > r) return null;
        normal = c.sub(v2).normalize();
        penetration = r - dist;
        contactPoint = v2;
      } else {
        normal = norms[edgeIndex];
        penetration = r - separation;
        contactPoint = c.sub(normal.mul(r));
      }
    }
    return { normal: normal.neg(), penetration, contacts: [contactPoint] };
    // normal points from polyBody -> circleBody convention handled by caller ordering
  }

  // Master dispatch: returns manifold with normal pointing from a -> b, or null
  function testCollision(a, b) {
    if (a.shape.type === 'circle' && b.shape.type === 'circle') {
      return circleVsCircle(a, b);
    }
    if (a.shape.type === 'polygon' && b.shape.type === 'polygon') {
      return polygonVsPolygon(a, b);
    }
    if (a.shape.type === 'circle' && b.shape.type === 'polygon') {
      const m = circleVsPolygon(a, b);
      if (!m) return null;
      return { normal: m.normal.neg(), penetration: m.penetration, contacts: m.contacts };
    }
    if (a.shape.type === 'polygon' && b.shape.type === 'circle') {
      const m = circleVsPolygon(b, a);
      if (!m) return null;
      return { normal: m.normal, penetration: m.penetration, contacts: m.contacts };
    }
    return null;
  }

  // Raycasting: ray vs circle, ray vs polygon
  function raycastCircle(origin, dir, body, maxDist) {
    const r = body.shape.radius;
    const m = origin.sub(body.pos);
    const b = m.dot(dir);
    const c = m.dot(m) - r * r;
    if (c > 0 && b > 0) return null;
    const disc = b * b - c;
    if (disc < 0) return null;
    let t = -b - Math.sqrt(disc);
    if (t < 0) t = 0;
    if (t > maxDist) return null;
    const point = origin.add(dir.mul(t));
    return { point, normal: point.sub(body.pos).normalize(), distance: t, body };
  }

  function raycastPolygon(origin, dir, body, maxDist) {
    const verts = body.getWorldVertices();
    const norms = body.getWorldNormals();
    let tMin = 0, tMax = maxDist, hitNormal = null;
    for (let i = 0; i < verts.length; i++) {
      const n = norms[i], v = verts[i];
      const denom = n.dot(dir);
      const dist = n.dot(v.sub(origin));
      if (Math.abs(denom) < 1e-9) {
        if (dist < 0) return null; // parallel and outside
      } else {
        const t = dist / denom;
        if (denom < 0) { if (t > tMin) { tMin = t; hitNormal = n; } }
        else { if (t < tMax) tMax = t; }
        if (tMin > tMax) return null;
      }
    }
    if (!hitNormal) return null;
    return { point: origin.add(dir.mul(tMin)), normal: hitNormal, distance: tMin, body };
  }

  function raycast(origin, dir, bodies, maxDist = Infinity) {
    let closest = null;
    const d = dir.normalize();
    for (const body of bodies) {
      const hit = body.shape.type === 'circle'
        ? raycastCircle(origin, d, body, maxDist)
        : raycastPolygon(origin, d, body, maxDist);
      if (hit && (!closest || hit.distance < closest.distance)) closest = hit;
    }
    return closest;
  }

  global.__PhysicsPartial.aabbOverlap = aabbOverlap;
  global.__PhysicsPartial.testCollision = testCollision;
  global.__PhysicsPartial.raycast = raycast;
})(typeof window !== 'undefined' ? window : globalThis);

// ==========================================================================
// Broad-phase: Spatial Hash Grid - avoids O(n^2) pair testing for large scenes
// ==========================================================================
(function (global) {
  'use strict';
  class SpatialHashGrid {
    constructor(cellSize = 64) {
      this.cellSize = cellSize;
      this.cells = new Map();
    }
    _key(x, y) { return x + ',' + y; }
    _cellsFor(aabb) {
      const cs = this.cellSize;
      const minX = Math.floor(aabb.minX / cs), maxX = Math.floor(aabb.maxX / cs);
      const minY = Math.floor(aabb.minY / cs), maxY = Math.floor(aabb.maxY / cs);
      const keys = [];
      for (let x = minX; x <= maxX; x++)
        for (let y = minY; y <= maxY; y++)
          keys.push(this._key(x, y));
      return keys;
    }
    clear() { this.cells.clear(); }
    insert(body) {
      const aabb = body.getAABB();
      for (const k of this._cellsFor(aabb)) {
        if (!this.cells.has(k)) this.cells.set(k, []);
        this.cells.get(k).push(body);
      }
    }
    getPotentialPairs() {
      const seen = new Set();
      const pairs = [];
      for (const bucket of this.cells.values()) {
        for (let i = 0; i < bucket.length; i++) {
          for (let j = i + 1; j < bucket.length; j++) {
            const a = bucket[i], b = bucket[j];
            const key = a.id < b.id ? a.id + '_' + b.id : b.id + '_' + a.id;
            if (seen.has(key)) continue;
            seen.add(key);
            pairs.push([a, b]);
          }
        }
      }
      return pairs;
    }
  }
  global.__PhysicsPartial.SpatialHashGrid = SpatialHashGrid;
})(typeof window !== 'undefined' ? window : globalThis);

// ==========================================================================
// Sequential-impulse contact solver: restitution, Coulomb friction, positional correction
// ==========================================================================
(function (global) {
  'use strict';
  const { Vec2 } = global.__PhysicsPartial;

  const SLOP = 0.01;           // allowed penetration before correcting (avoids jitter)
  const CORRECTION_PERCENT = 0.8; // Baumgarte stabilization percentage

  class Contact {
    constructor(a, b, manifold) {
      this.a = a; this.b = b;
      this.normal = manifold.normal;   // points from a -> b
      this.penetration = manifold.penetration;
      this.contacts = manifold.contacts;
      this.restitution = Math.max(a.restitution, b.restitution);
      this.friction = Math.sqrt(a.staticFriction * b.staticFriction);
      this.dynamicFriction = Math.sqrt(a.dynamicFriction * b.dynamicFriction);
    }

    // resolve using sequential impulses at each contact point
    resolve() {
      const { a, b, normal } = this;
      const invMassSum = a.invMass + b.invMass;
      if (invMassSum === 0) return;

      for (const point of this.contacts) {
        const ra = point.sub(a.pos);
        const rb = point.sub(b.pos);

        const relVel = b.velocityAtPoint(point).sub(a.velocityAtPoint(point));
        const velAlongNormal = relVel.dot(normal);
        if (velAlongNormal > 0) continue; // separating already

        const raCrossN = ra.cross(normal);
        const rbCrossN = rb.cross(normal);
        const denom = invMassSum + a.invInertia * raCrossN * raCrossN + b.invInertia * rbCrossN * rbCrossN;
        if (denom < 1e-9) continue;

        let j = -(1 + this.restitution) * velAlongNormal / denom;
        j /= this.contacts.length;

        const impulse = normal.mul(j);
        a.applyImpulse(impulse.neg(), point);
        b.applyImpulse(impulse, point);

        // --- Coulomb friction (Amonton's law): |Ff| <= mu * Fn ---
        const relVel2 = b.velocityAtPoint(point).sub(a.velocityAtPoint(point));
        let tangent = relVel2.sub(normal.mul(relVel2.dot(normal)));
        const tLen = tangent.length();
        if (tLen > 1e-9) {
          tangent = tangent.div(tLen);
          const raCrossT = ra.cross(tangent);
          const rbCrossT = rb.cross(tangent);
          const denomT = invMassSum + a.invInertia * raCrossT * raCrossT + b.invInertia * rbCrossT * rbCrossT;
          let jt = -relVel2.dot(tangent) / (denomT || 1e-9);
          jt /= this.contacts.length;

          let frictionImpulse;
          if (Math.abs(jt) < j * this.friction) {
            frictionImpulse = tangent.mul(jt); // within static friction cone
          } else {
            frictionImpulse = tangent.mul(-j * this.dynamicFriction); // sliding (kinetic)
          }
          a.applyImpulse(frictionImpulse.neg(), point);
          b.applyImpulse(frictionImpulse, point);
        }
      }
    }

    // positional correction to prevent sinking objects (Baumgarte stabilization)
    correctPositions() {
      const { a, b, normal, penetration } = this;
      const invMassSum = a.invMass + b.invMass;
      if (invMassSum === 0) return;
      const correctionMag = Math.max(penetration - SLOP, 0) / invMassSum * CORRECTION_PERCENT;
      const correction = normal.mul(correctionMag);
      if (!a.isStatic) a.pos.subEq(correction.mul(a.invMass));
      if (!b.isStatic) b.pos.addEq(correction.mul(b.invMass));
    }
  }

  global.__PhysicsPartial.Contact = Contact;
})(typeof window !== 'undefined' ? window : globalThis);

// ==========================================================================
// Joints connecting two RigidBodies
// ==========================================================================
(function (global) {
  'use strict';
  const { Vec2 } = global.__PhysicsPartial;

  // Rigid distance joint (keeps two bodies a fixed distance apart, like a rod)
  class DistanceJoint {
    constructor(a, b, opts = {}) {
      this.a = a; this.b = b;
      this.anchorA = opts.anchorA ? opts.anchorA.clone() : a.pos.clone();
      this.anchorB = opts.anchorB ? opts.anchorB.clone() : b.pos.clone();
      this.length = opts.length !== undefined ? opts.length : a.pos.distanceTo(b.pos);
      this.stiffness = opts.stiffness !== undefined ? opts.stiffness : 1;
    }
    solve() {
      const { a, b } = this;
      const pa = a.pos.add(this.anchorA.sub(a.pos));
      const pb = b.pos.add(this.anchorB.sub(b.pos));
      const delta = pb.sub(pa);
      const dist = delta.length() || 1e-6;
      const diff = (dist - this.length) / dist * this.stiffness;
      const invSum = a.invMass + b.invMass;
      if (invSum === 0) return;
      const correction = delta.mul(diff / invSum);
      if (!a.isStatic) a.pos.addEq(correction.mul(a.invMass));
      if (!b.isStatic) b.pos.subEq(correction.mul(b.invMass));
    }
  }

  // Spring joint (Hooke's law force between two bodies, no hard constraint)
  class SpringJoint {
    constructor(a, b, opts = {}) {
      this.a = a; this.b = b;
      this.length = opts.length !== undefined ? opts.length : a.pos.distanceTo(b.pos);
      this.k = opts.stiffness !== undefined ? opts.stiffness : 100;
      this.damping = opts.damping !== undefined ? opts.damping : 5;
    }
    apply() {
      const { a, b } = this;
      const delta = b.pos.sub(a.pos);
      const dist = delta.length() || 1e-6;
      const dir = delta.div(dist);
      const stretch = dist - this.length;
      const relVel = b.velocity.sub(a.velocity).dot(dir);
      const forceMag = this.k * stretch + this.damping * relVel;
      const force = dir.mul(forceMag);
      a.applyForce(force);
      b.applyForce(force.neg());
    }
  }

  // Revolute (pin) joint: pins a local point on each body together, allows free rotation
  class RevoluteJoint {
    constructor(a, b, worldAnchor, opts = {}) {
      this.a = a; this.b = b;
      this.localAnchorA = worldAnchor.sub(a.pos).rotate(-a.angle);
      this.localAnchorB = worldAnchor.sub(b.pos).rotate(-b.angle);
      this.beta = opts.beta !== undefined ? opts.beta : 0.2;
    }
    solve() {
      const { a, b } = this;
      const rA = this.localAnchorA.rotate(a.angle);
      const rB = this.localAnchorB.rotate(b.angle);
      const worldA = a.pos.add(rA);
      const worldB = b.pos.add(rB);
      const C = worldB.sub(worldA); // constraint violation

      const invMassSum = a.invMass + b.invMass;
      if (invMassSum === 0) return;

      // relative velocity at anchor
      const relVel = b.velocityAtPoint(worldB).sub(a.velocityAtPoint(worldA));

      // Solve for impulse (2x2 effective mass, approximated per-axis for simplicity/perf)
      for (const axis of [new Vec2(1, 0), new Vec2(0, 1)]) {
        const raCrossN = rA.cross(axis);
        const rbCrossN = rB.cross(axis);
        const denom = invMassSum + a.invInertia * raCrossN * raCrossN + b.invInertia * rbCrossN * rbCrossN;
        if (denom < 1e-9) continue;
        const bias = this.beta * axis.dot(C);
        const j = -(relVel.dot(axis) + bias) / denom;
        const impulse = axis.mul(j);
        a.applyImpulse(impulse.neg(), worldA);
        b.applyImpulse(impulse, worldB);
      }
    }
  }

  // Motor joint: drives a body's angular velocity toward a target (e.g. wheels, fans)
  class MotorJoint {
    constructor(body, opts = {}) {
      this.body = body;
      this.targetSpeed = opts.speed || 0;
      this.maxTorque = opts.maxTorque !== undefined ? opts.maxTorque : 1000;
    }
    apply(dt) {
      const b = this.body;
      const speedDiff = this.targetSpeed - b.angularVelocity;
      let torque = speedDiff * b.inertia / dt;
      torque = Math.max(-this.maxTorque, Math.min(this.maxTorque, torque));
      b.applyTorque(torque);
    }
  }

  global.__PhysicsPartial.DistanceJoint = DistanceJoint;
  global.__PhysicsPartial.SpringJoint = SpringJoint;
  global.__PhysicsPartial.RevoluteJoint = RevoluteJoint;
  global.__PhysicsPartial.MotorJoint = MotorJoint;
})(typeof window !== 'undefined' ? window : globalThis);

// ==========================================================================
// SoftBody - mass-spring mesh (Verlet particles + distance constraints)
// Includes ready-made factories for cloth, rope, and pressurized "blob" bodies.
// ==========================================================================
(function (global) {
  'use strict';
  const { Vec2, Particle, DistanceConstraint } = global.__PhysicsPartial;

  class SoftBody {
    constructor() {
      this.particles = [];
      this.constraints = [];
      this.pressure = 0; // if >0, acts as an internal-pressure "blob" (keeps particles pushed outward)
      this.usePressure = false;
    }

    addParticle(p) { this.particles.push(p); return p; }
    addConstraint(c) { this.constraints.push(c); return c; }

    // Rope of N particles connected by distance constraints; first particle pinned at (x,y)
    static rope(x, y, segmentLength, segments, opts = {}) {
      const sb = new SoftBody();
      let prev = null;
      for (let i = 0; i < segments; i++) {
        const p = new Particle(x, y + i * segmentLength, { mass: opts.mass || 1, damping: opts.damping });
        if (i === 0 && opts.pinStart !== false) p.pin();
        sb.addParticle(p);
        if (prev) sb.addConstraint(new DistanceConstraint(prev, p, { length: segmentLength, stiffness: opts.stiffness || 1 }));
        prev = p;
      }
      return sb;
    }

    // Rectangular cloth grid, corners optionally pinned
    static cloth(x, y, cols, rows, spacing, opts = {}) {
      const sb = new SoftBody();
      const grid = [];
      for (let r = 0; r < rows; r++) {
        grid[r] = [];
        for (let c = 0; c < cols; c++) {
          const p = new Particle(x + c * spacing, y + r * spacing, { mass: opts.mass || 1, damping: opts.damping });
          if (r === 0 && (opts.pinTop !== false)) p.pin();
          grid[r][c] = p;
          sb.addParticle(p);
          if (c > 0) sb.addConstraint(new DistanceConstraint(grid[r][c - 1], p, { stiffness: opts.stiffness || 1 }));
          if (r > 0) sb.addConstraint(new DistanceConstraint(grid[r - 1][c], p, { stiffness: opts.stiffness || 1 }));
          // shear constraints (diagonals) for structural stability
          if (r > 0 && c > 0 && opts.shear !== false) {
            sb.addConstraint(new DistanceConstraint(grid[r - 1][c - 1], p, { stiffness: (opts.stiffness || 1) * 0.5 }));
            sb.addConstraint(new DistanceConstraint(grid[r][c - 1], grid[r - 1][c], { stiffness: (opts.stiffness || 1) * 0.5 }));
          }
        }
      }
      sb.grid = grid;
      return sb;
    }

    // Soft circular "blob" using a ring of particles + internal pressure force (like a balloon)
    static blob(x, y, radius, numPoints, opts = {}) {
      const sb = new SoftBody();
      for (let i = 0; i < numPoints; i++) {
        const a = (i / numPoints) * Math.PI * 2;
        const p = new Particle(x + Math.cos(a) * radius, y + Math.sin(a) * radius, { mass: opts.mass || 1, damping: opts.damping });
        sb.addParticle(p);
      }
      for (let i = 0; i < numPoints; i++) {
        const a = sb.particles[i], b = sb.particles[(i + 1) % numPoints];
        sb.addConstraint(new DistanceConstraint(a, b, { stiffness: opts.stiffness !== undefined ? opts.stiffness : 0.9 }));
      }
      // spokes to a virtual center for shape retention (soft, low stiffness so it squishes)
      if (opts.spokes !== false) {
        sb.center = new Particle(x, y, { mass: (opts.mass || 1) * numPoints * 0.5, damping: opts.damping });
        sb.addParticle(sb.center);
        for (const p of sb.particles.slice(0, numPoints)) {
          sb.addConstraint(new DistanceConstraint(sb.center, p, { length: radius, stiffness: 0.15 }));
        }
      }
      sb.usePressure = opts.usePressure !== false;
      sb.pressure = opts.pressure !== undefined ? opts.pressure : 2000;
      sb.restArea = Math.PI * radius * radius;
      sb.ring = sb.particles.slice(0, numPoints);
      return sb;
    }

    // apply internal gas-pressure force: F = pressure * (restArea/currentArea) * edgeNormal * edgeLength
    _applyPressure() {
      if (!this.usePressure || !this.ring) return;
      const ring = this.ring;
      let area = 0;
      for (let i = 0; i < ring.length; i++) {
        const a = ring[i].pos, b = ring[(i + 1) % ring.length].pos;
        area += a.cross(b);
      }
      area = Math.abs(area) * 0.5;
      const ratio = this.restArea / Math.max(area, 1e-6);
      for (let i = 0; i < ring.length; i++) {
        const a = ring[i], b = ring[(i + 1) % ring.length];
        const edge = b.pos.sub(a.pos);
        const normal = edge.rightNormal().normalize();
        const edgeLen = edge.length();
        const force = normal.mul(this.pressure * ratio * edgeLen * 0.5);
        a.applyForce(force);
        b.applyForce(force);
      }
    }

    step(dt, gravity, iterations) {
      this._applyPressure();
      for (const p of this.particles) {
        if (!p.pinned) p.applyForce(gravity.mul(p.mass));
        p.integrate(dt);
      }
      for (let i = 0; i < iterations; i++) {
        for (const c of this.constraints) c.solve();
      }
    }

    // simple particle-vs-circle-body collision (push particles out of solid circles)
    collideWithCircle(circlePos, radius, restitution = 0.3) {
      for (const p of this.particles) {
        const delta = p.pos.sub(circlePos);
        const dist = delta.length();
        const minDist = radius + p.radius;
        if (dist < minDist && dist > 1e-6) {
          const n = delta.div(dist);
          p.pos.copy(circlePos.add(n.mul(minDist)));
          const vel = p.velocity;
          const vDotN = vel.dot(n);
          if (vDotN < 0) p.velocity = vel.sub(n.mul(vDotN * (1 + restitution)));
        }
      }
    }

    // constrain all particles inside an axis-aligned box (e.g. the canvas) with bounce
    collideWithBounds(minX, minY, maxX, maxY) {
      for (const p of this.particles) {
        const r = p.radius;
        const vel = p.velocity;
        if (p.pos.x - r < minX) { p.pos.x = minX + r; p.velocity = new Vec2(-vel.x * p.restitution, vel.y); }
        if (p.pos.x + r > maxX) { p.pos.x = maxX - r; p.velocity = new Vec2(-vel.x * p.restitution, vel.y); }
        if (p.pos.y - r < minY) { p.pos.y = minY + r; p.velocity = new Vec2(vel.x, -vel.y * p.restitution); }
        if (p.pos.y + r > maxY) { p.pos.y = maxY - r; p.velocity = new Vec2(vel.x, -vel.y * p.restitution); }
      }
    }
  }

  global.__PhysicsPartial.SoftBody = SoftBody;
})(typeof window !== 'undefined' ? window : globalThis);

// ==========================================================================
// World - the main simulation container. Owns bodies, particles/soft-bodies,
// joints, and runs the full step pipeline each frame.
// ==========================================================================
(function (global) {
  'use strict';
  const P = global.__PhysicsPartial;
  const { Vec2, SpatialHashGrid, aabbOverlap, testCollision, Contact, raycast } = P;

  const SLEEP_LINEAR_THRESHOLD = 0.02;
  const SLEEP_ANGULAR_THRESHOLD = 0.02;
  const SLEEP_TIME = 0.6; // seconds of stillness before sleeping

  class World {
    constructor(opts = {}) {
      this.gravity = opts.gravity ? new Vec2(opts.gravity.x, opts.gravity.y) : new Vec2(0, 980); // px/s^2, "down" positive
      this.bodies = [];
      this.softBodies = [];
      this.joints = [];
      this.grid = new SpatialHashGrid(opts.cellSize || 80);
      this.solverIterations = opts.solverIterations !== undefined ? opts.solverIterations : 8;
      this.constraintIterations = opts.constraintIterations !== undefined ? opts.constraintIterations : 6;
      this.substeps = opts.substeps !== undefined ? opts.substeps : 1;
      this.bounds = opts.bounds || null; // { minX, minY, maxX, maxY } optional world boundary
      this.boundsRestitution = opts.boundsRestitution !== undefined ? opts.boundsRestitution : 0.6;
      this.enableSleeping = opts.enableSleeping !== undefined ? opts.enableSleeping : true;
      this.contacts = [];
      this.listeners = { collisionStart: [], collisionEnd: [] };
      this._activePairsLastFrame = new Set();
    }

    addBody(b) { this.bodies.push(b); return b; }
    removeBody(b) { const i = this.bodies.indexOf(b); if (i >= 0) this.bodies.splice(i, 1); }
    addSoftBody(sb) { this.softBodies.push(sb); return sb; }
    addJoint(j) { this.joints.push(j); return j; }
    removeJoint(j) { const i = this.joints.indexOf(j); if (i >= 0) this.joints.splice(i, 1); }

    on(event, cb) { if (this.listeners[event]) this.listeners[event].push(cb); }

    raycast(origin, dir, maxDist) { return raycast(origin, dir, this.bodies, maxDist); }

    step(dt) {
      const sub = this.substeps;
      const h = dt / sub;
      for (let s = 0; s < sub; s++) this._step(h);
    }

    _step(dt) {
      // 1. Apply joints (springs act as forces, distance/revolute as constraints solved later)
      for (const j of this.joints) {
        if (j.apply) j.apply(dt);
      }

      // 2. Integrate rigid bodies (semi-implicit Euler)
      for (const b of this.bodies) {
        b.integrate(dt, this.gravity);
      }

      // 3. Solve position-based joints (distance/revolute) iteratively
      for (let i = 0; i < this.constraintIterations; i++) {
        for (const j of this.joints) {
          if (j.solve) j.solve();
        }
      }

      // 4. Broad-phase: build spatial grid, gather candidate pairs
      this.grid.clear();
      const dynamicBodies = [];
      for (const b of this.bodies) {
        if (b.sleeping) continue;
        this.grid.insert(b);
        if (!b.isStatic) dynamicBodies.push(b);
      }
      const pairs = this.grid.getPotentialPairs();

      // 5. Narrow-phase: precise collision test + build contacts
      const newContacts = [];
      const activePairs = new Set();
      for (const [a, b] of pairs) {
        if (a.isStatic && b.isStatic) continue;
        if (a.sleeping && b.sleeping) continue;
        if (a.sleeping && b.isStatic) continue;
        if (b.sleeping && a.isStatic) continue;
        if ((a.categoryBits & b.maskBits) === 0 || (b.categoryBits & a.maskBits) === 0) continue;
        if (a.collisionGroup !== 0 && a.collisionGroup === b.collisionGroup) continue;
        if (!aabbOverlap(a.getAABB(), b.getAABB())) continue;

        const manifold = testCollision(a, b);
        if (!manifold) continue;

        if (a.sleeping) a.wake();
        if (b.sleeping) b.wake();

        const pairKey = a.id + '_' + b.id;
        activePairs.add(pairKey);
        if (!this._activePairsLastFrame.has(pairKey)) {
          this._fire('collisionStart', a, b, manifold);
        }

        if (a.isSensor || b.isSensor) continue; // sensors detect but don't resolve

        newContacts.push(new Contact(a, b, manifold));
      }
      for (const key of this._activePairsLastFrame) {
        if (!activePairs.has(key)) {
          const [aid, bid] = key.split('_').map(Number);
          const a = this.bodies.find(bd => bd.id === aid);
          const b = this.bodies.find(bd => bd.id === bid);
          this._fire('collisionEnd', a, b, null);
        }
      }
      this._activePairsLastFrame = activePairs;
      this.contacts = newContacts;

      // 6. Solve velocity constraints (impulses) - multiple iterations for stability/accuracy
      for (let i = 0; i < this.solverIterations; i++) {
        for (const c of newContacts) c.resolve();
      }

      // 7. Positional correction (fixes residual penetration / sinking)
      for (const c of newContacts) c.correctPositions();

      // 8. World bounds (optional simple box container)
      if (this.bounds) this._resolveBounds();

      // 9. Sleeping
      if (this.enableSleeping) this._updateSleep(dt);

      // 10. Soft bodies (independent particle simulation)
      for (const sb of this.softBodies) {
        sb.step(dt, this.gravity, this.constraintIterations);
        if (this.bounds) sb.collideWithBounds(this.bounds.minX, this.bounds.minY, this.bounds.maxX, this.bounds.maxY);
      }
    }

    _fire(event, a, b, manifold) {
      for (const cb of this.listeners[event]) cb(a, b, manifold);
    }

    _resolveBounds() {
      const { minX, minY, maxX, maxY } = this.bounds;
      for (const b of this.bodies) {
        if (b.isStatic || b.sleeping) continue;
        const aabb = b.getAABB();
        const halfW = (aabb.maxX - aabb.minX) / 2, halfH = (aabb.maxY - aabb.minY) / 2;
        if (aabb.minX < minX) { b.pos.x += minX - aabb.minX; if (b.velocity.x < 0) b.velocity.x *= -this.boundsRestitution; }
        if (aabb.maxX > maxX) { b.pos.x -= aabb.maxX - maxX; if (b.velocity.x > 0) b.velocity.x *= -this.boundsRestitution; }
        if (aabb.minY < minY) { b.pos.y += minY - aabb.minY; if (b.velocity.y < 0) b.velocity.y *= -this.boundsRestitution; }
        if (aabb.maxY > maxY) { b.pos.y -= aabb.maxY - maxY; if (b.velocity.y > 0) b.velocity.y *= -this.boundsRestitution; }
      }
    }

    _updateSleep(dt) {
      for (const b of this.bodies) {
        if (b.isStatic || !b.allowSleep) continue;
        const slow = b.velocity.lengthSq() < SLEEP_LINEAR_THRESHOLD * SLEEP_LINEAR_THRESHOLD &&
                     Math.abs(b.angularVelocity) < SLEEP_ANGULAR_THRESHOLD;
        if (slow) {
          b.sleepTimer += dt;
          if (b.sleepTimer > SLEEP_TIME) { b.sleeping = true; b.velocity.set(0, 0); b.angularVelocity = 0; }
        } else {
          b.sleepTimer = 0;
        }
      }
    }
  }

  P.World = World;
})(typeof window !== 'undefined' ? window : globalThis);

// ==========================================================================
// Debug / convenience Canvas2D renderer (optional - purely for visualizing)
// ==========================================================================
(function (global) {
  'use strict';
  function renderWorld(ctx, world, opts = {}) {
    ctx.save();
    for (const b of world.bodies) {
      ctx.save();
      ctx.translate(b.pos.x, b.pos.y);
      ctx.rotate(b.angle);
      ctx.beginPath();
      if (b.shape.type === 'circle') {
        ctx.arc(0, 0, b.shape.radius, 0, Math.PI * 2);
        ctx.moveTo(0, 0); ctx.lineTo(b.shape.radius, 0); // spoke to show rotation
      } else {
        const v = b.shape.vertices;
        ctx.moveTo(v[0].x, v[0].y);
        for (let i = 1; i < v.length; i++) ctx.lineTo(v[i].x, v[i].y);
        ctx.closePath();
      }
      ctx.fillStyle = b.sleeping ? (opts.sleepColor || 'rgba(120,120,120,0.5)') : (b.isStatic ? (opts.staticColor || '#555') : (opts.bodyColor || 'rgba(80,160,255,0.7)'));
      ctx.fill();
      ctx.strokeStyle = opts.strokeColor || '#222';
      ctx.stroke();
      ctx.restore();
    }
    for (const sb of world.softBodies) {
      ctx.strokeStyle = opts.softBodyColor || '#2ecc71';
      ctx.lineWidth = 1.5;
      for (const c of sb.constraints) {
        if (c.broken) continue;
        ctx.beginPath();
        ctx.moveTo(c.a.pos.x, c.a.pos.y);
        ctx.lineTo(c.b.pos.x, c.b.pos.y);
        ctx.stroke();
      }
      for (const p of sb.particles) {
        ctx.beginPath();
        ctx.arc(p.pos.x, p.pos.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = p.pinned ? '#e74c3c' : (opts.particleColor || '#27ae60');
        ctx.fill();
      }
    }
    ctx.restore();
  }
  global.__PhysicsPartial.renderWorld = renderWorld;
})(typeof window !== 'undefined' ? window : globalThis);

// ==========================================================================
// Public API export
// ==========================================================================
(function (global) {
  'use strict';
  const P = global.__PhysicsPartial;
  const Physics = {
    Vec2: P.Vec2,
    Particle: P.Particle,
    DistanceConstraint: P.DistanceConstraint,
    SpringConstraint: P.SpringConstraint,
    PinConstraint: P.PinConstraint,
    AngleConstraint: P.AngleConstraint,
    CircleShape: P.CircleShape,
    PolygonShape: P.PolygonShape,
    RigidBody: P.RigidBody,
    SpatialHashGrid: P.SpatialHashGrid,
    Contact: P.Contact,
    DistanceJoint: P.DistanceJoint,
    SpringJoint: P.SpringJoint,
    RevoluteJoint: P.RevoluteJoint,
    MotorJoint: P.MotorJoint,
    SoftBody: P.SoftBody,
    World: P.World,
    raycast: P.raycast,
    renderWorld: P.renderWorld,
    clamp: P.clamp,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = Physics;
  } else {
    global.Physics = Physics;
  }
  delete global.__PhysicsPartial;
})(typeof window !== 'undefined' ? window : globalThis);

/*
 * ============================ QUICK-START DEMO ============================
 *
 * <canvas id="c" width="800" height="600"></canvas>
 * <script src="physics.js"></script>
 * <script>
 *   const canvas = document.getElementById('c');
 *   const ctx = canvas.getContext('2d');
 *
 *   const world = new Physics.World({
 *     gravity: { x: 0, y: 980 },
 *     bounds: { minX: 0, minY: 0, maxX: canvas.width, maxY: canvas.height },
 *     boundsRestitution: 0.7,
 *   });
 *
 *   // ground
 *   world.addBody(Physics.RigidBody.box(400, 590, 800, 20, { isStatic: true }));
 *
 *   // bouncy balls
 *   for (let i = 0; i < 10; i++) {
 *     world.addBody(Physics.RigidBody.circle(100 + i * 60, 50, 20, { restitution: 0.85, dynamicFriction: 0.2 }));
 *   }
 *
 *   // tumbling boxes
 *   for (let i = 0; i < 5; i++) {
 *     world.addBody(Physics.RigidBody.box(200 + i * 50, 100, 40, 40, { restitution: 0.3, angle: Math.random() }));
 *   }
 *
 *   // soft-body cloth pinned at the top
 *   world.addSoftBody(Physics.SoftBody.cloth(250, 20, 12, 8, 20, { stiffness: 0.9 }));
 *
 *   // a bouncy pressurized blob
 *   world.addSoftBody(Physics.SoftBody.blob(600, 150, 40, 16, { pressure: 3000 }));
 *
 *   let last = performance.now();
 *   function loop(now) {
 *     const dt = Math.min((now - last) / 1000, 1 / 30);
 *     last = now;
 *     world.step(dt);
 *     ctx.clearRect(0, 0, canvas.width, canvas.height);
 *     Physics.renderWorld(ctx, world);
 *     requestAnimationFrame(loop);
 *   }
 *   requestAnimationFrame(loop);
 * </script>
 * ===========================================================================
 */
