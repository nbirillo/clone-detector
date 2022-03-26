package org.jetbrains.research.cloneDetector.core.structures

interface IndexedSequence {
    /**
     * Sequence to be indexed
     */
    val sequence: Sequence<SourceToken>
}
