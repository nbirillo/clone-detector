package org.jetbrains.research.cloneDetector.core.structures

import org.jetbrains.research.cloneDetector.core.utils.tokenSequence

class RangeCloneClass(val cloneRanges: List<Clone>) : CloneClass {
    init {
        require(cloneRanges.size > 1)
    }

    override val size: Int
        get() = cloneRanges.size

    override val clones: Sequence<Clone>
        get() = cloneRanges.asSequence()

    override val length: Int by lazy {
        cloneRanges[0].tokenSequence().count()
    }
}
