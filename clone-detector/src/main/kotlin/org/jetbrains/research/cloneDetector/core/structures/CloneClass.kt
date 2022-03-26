package org.jetbrains.research.cloneDetector.core.structures

interface CloneClass {
    val clones: Sequence<Clone>
    val size: Int
    val length: Int
}
