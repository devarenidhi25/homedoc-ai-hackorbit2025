"use client"

import { useEffect, useRef } from "react"

const useScrollReveal = () => {
  const elementsRef = useRef([])

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("revealed")
          }
        })
      },
      {
        threshold: 0.1,
        rootMargin: "0px 0px -50px 0px",
      },
    )

    const elements = document.querySelectorAll(".scroll-reveal")
    elements.forEach((el) => observer.observe(el))
    elementsRef.current = elements

    return () => {
      elementsRef.current.forEach((el) => observer.unobserve(el))
    }
  }, [])

  return elementsRef
}

export default useScrollReveal