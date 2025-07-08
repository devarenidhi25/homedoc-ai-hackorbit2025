"use client"

import { useEffect, useRef } from "react"
import * as THREE from "three"

const Three3DBackground = () => {
  const mountRef = useRef(null)
  const sceneRef = useRef(null)
  const rendererRef = useRef(null)
  const frameRef = useRef(null)

  useEffect(() => {
    if (!mountRef.current) return

    // Scene setup
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000)
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })

    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setClearColor(0x000000, 0)
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap

    mountRef.current.appendChild(renderer.domElement)
    sceneRef.current = scene
    rendererRef.current = renderer

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0x8b5cf6, 0.8) // Purple light
    directionalLight.position.set(10, 10, 5)
    directionalLight.castShadow = true
    scene.add(directionalLight)

    const pointLight1 = new THREE.PointLight(0x06b6d4, 0.6, 100) // Cyan light
    pointLight1.position.set(-20, 10, 10)
    scene.add(pointLight1)

    const pointLight2 = new THREE.PointLight(0x10b981, 0.5, 100) // Green light
    pointLight2.position.set(20, -10, 5)
    scene.add(pointLight2)

    // Medical-themed geometries
    const geometries = []
    const materials = []
    const meshes = []

    // DNA Helix
    const helixGeometry = new THREE.CylinderGeometry(0.1, 0.1, 10, 8);
    const helixMaterial = new THREE.MeshPhongMaterial({
      color: 0xff0000, // Red color
      transparent: true,
      opacity: 0.15, // Still semi-transparent
      shininess: 100,
    });

    const helix1 = new THREE.Mesh(helixGeometry, helixMaterial)
    const helix2 = new THREE.Mesh(helixGeometry, helixMaterial)

    helix1.position.set(-15, 0, -20)
    helix2.position.set(-15, 0, -20)
    helix2.rotation.z = Math.PI / 2

    scene.add(helix1)
    scene.add(helix2)
    geometries.push(helixGeometry)
    materials.push(helixMaterial)
    meshes.push(helix1, helix2)

    // Floating Pills
    for (let i = 0; i < 20; i++) {
      const pillGeometry = new THREE.CapsuleGeometry(0.3, 1.5, 4, 8)
      const pillMaterial = new THREE.MeshPhongMaterial({
        color: Math.random() > 0.5 ? 0x06b6d4 : 0xff0000, // Cyan and purple
        transparent: true,
        opacity: 0.7,
        shininess: 100,
      })
      const pill = new THREE.Mesh(pillGeometry, pillMaterial)

      pill.position.set((Math.random() - 0.5) * 50, (Math.random() - 0.5) * 20, (Math.random() - 0.5) * 50)
      pill.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI)

      scene.add(pill)
      geometries.push(pillGeometry)
      materials.push(pillMaterial)
      meshes.push(pill)
    }

    // Molecular Structures
    for (let i = 0; i < 15; i++) {
      const sphereGeometry = new THREE.SphereGeometry(0.5, 16, 16)
      const sphereMaterial = new THREE.MeshPhongMaterial({
        color: 0x10b981, // Emerald green
        transparent: true,
        opacity: 0.6,
        shininess: 100,
      })
      const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial)

      sphere.position.set((Math.random() - 0.5) * 60, (Math.random() - 0.5) * 30, (Math.random() - 0.5) * 60)

      scene.add(sphere)
      geometries.push(sphereGeometry)
      materials.push(sphereMaterial)
      meshes.push(sphere)
    }

    // Heart Beat Rings
    for (let i = 0; i < 8; i++) {
      const ringGeometry = new THREE.RingGeometry(2, 2.5, 32)
      const ringMaterial = new THREE.MeshBasicMaterial({
        color: 0xf59e0b, // Amber color
        transparent: true,
        opacity: 0.25,
        side: THREE.DoubleSide,
      })
      const ring = new THREE.Mesh(ringGeometry, ringMaterial)

      ring.position.set((Math.random() - 0.5) * 40, (Math.random() - 0.5) * 20, (Math.random() - 0.5) * 40)
      ring.rotation.x = Math.random() * Math.PI
      ring.rotation.y = Math.random() * Math.PI

      scene.add(ring)
      geometries.push(ringGeometry)
      materials.push(ringMaterial)
      meshes.push(ring)
    }

    // Stethoscope-like Curves
    const curvePoints = []
    for (let i = 0; i < 100; i++) {
      const angle = (i / 100) * Math.PI * 4
      curvePoints.push(new THREE.Vector3(Math.cos(angle) * 5, Math.sin(angle * 2) * 2, Math.sin(angle) * 5))
    }

    const curve = new THREE.CatmullRomCurve3(curvePoints)
    const tubeGeometry = new THREE.TubeGeometry(curve, 100, 0.1, 8, false)
    const tubeMaterial = new THREE.MeshPhongMaterial({
      color: 0x06b6d4, // Cyan color
      transparent: true,
      opacity: 0.6,
    })
    const tube = new THREE.Mesh(tubeGeometry, tubeMaterial)
    tube.position.set(10, 0, -10)

    scene.add(tube)
    geometries.push(tubeGeometry)
    materials.push(tubeMaterial)
    meshes.push(tube)

    // Particle System
    const particleCount = 1000
    const particleGeometry = new THREE.BufferGeometry()
    const particlePositions = new Float32Array(particleCount * 3)
    const particleColors = new Float32Array(particleCount * 3)

    for (let i = 0; i < particleCount; i++) {
      particlePositions[i * 3] = (Math.random() - 0.5) * 100
      particlePositions[i * 3 + 1] = (Math.random() - 0.5) * 100
      particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 100

      const color = new THREE.Color()
      color.setHSL(0.5 + Math.random() * 0.3, 0.7, 0.6 + Math.random() * 0.2) // Cooler colors
      particleColors[i * 3] = color.r
      particleColors[i * 3 + 1] = color.g
      particleColors[i * 3 + 2] = color.b
    }

    particleGeometry.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3))
    particleGeometry.setAttribute("color", new THREE.BufferAttribute(particleColors, 3))

    const particleMaterial = new THREE.PointsMaterial({
      size: 0.5,
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
    })

    const particles = new THREE.Points(particleGeometry, particleMaterial)
    scene.add(particles)

    geometries.push(particleGeometry)
    materials.push(particleMaterial)

    // Camera position
    camera.position.set(0, 5, 20)
    camera.lookAt(0, 0, 0)

    // Mouse interaction
    const mouse = new THREE.Vector2()
    const onMouseMove = (event) => {
      mouse.x = (event.clientX / window.innerWidth) * 2 - 1
      mouse.y = -(event.clientY / window.innerHeight) * 2 + 1
    }
    window.addEventListener("mousemove", onMouseMove)

    // Animation loop
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate)

      const time = Date.now() * 0.001

      // Rotate DNA helix
      helix1.rotation.y = time * 0.5
      helix2.rotation.y = time * 0.5
      helix2.rotation.x = time * 0.3

      // Animate pills
      meshes.forEach((mesh, index) => {
        if (mesh.geometry.type === "CapsuleGeometry") {
          mesh.rotation.x += 0.01
          mesh.rotation.y += 0.02
          mesh.position.y += Math.sin(time + index) * 0.01
        }
      })

      // Animate molecules
      meshes.forEach((mesh, index) => {
        if (mesh.geometry.type === "SphereGeometry") {
          mesh.position.y += Math.sin(time * 2 + index) * 0.005
          mesh.rotation.x += 0.005
          mesh.rotation.y += 0.01
        }
      })

      // Animate rings (heartbeat effect)
      meshes.forEach((mesh, index) => {
        if (mesh.geometry.type === "RingGeometry") {
          const scale = 1 + Math.sin(time * 3 + index) * 0.2
          mesh.scale.set(scale, scale, scale)
          mesh.rotation.z += 0.01
        }
      })

      // Animate tube
      tube.rotation.y = time * 0.2
      tube.position.y = Math.sin(time) * 2

      // Animate particles
      const positions = particles.geometry.attributes.position.array
      for (let i = 0; i < particleCount; i++) {
        positions[i * 3 + 1] += Math.sin(time + i * 0.01) * 0.01
      }
      particles.geometry.attributes.position.needsUpdate = true
      particles.rotation.y = time * 0.1

      // Camera movement based on mouse
      camera.position.x += (mouse.x * 5 - camera.position.x) * 0.05
      camera.position.y += (-mouse.y * 5 - camera.position.y) * 0.05
      camera.lookAt(0, 0, 0)

      renderer.render(scene, camera)
    }

    animate()

    // Handle resize
    const handleResize = () => {
      if (!camera || !renderer) return
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
    }
    window.addEventListener("resize", handleResize)

    // Cleanup
    return () => {
      window.removeEventListener("mousemove", onMouseMove)
      window.removeEventListener("resize", handleResize)

      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current)
      }

      // Dispose of geometries and materials
      geometries.forEach((geometry) => geometry.dispose())
      materials.forEach((material) => material.dispose())

      if (rendererRef.current && mountRef.current) {
        mountRef.current.removeChild(rendererRef.current.domElement)
        rendererRef.current.dispose()
      }
    }
  }, [])

  return (
    <div
      ref={mountRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        zIndex: 0,
        pointerEvents: "none",
      }}
    />
  )
}

export default Three3DBackground
