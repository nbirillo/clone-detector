package org.jetbrains.research.cloneDetector.core.postprocessing

import org.jetbrains.research.cloneDetector.core.structures.CloneClass
import org.jetbrains.research.cloneDetector.core.structures.RangeCloneClass
import org.jetbrains.research.cloneDetector.core.structures.TreeCloneClass

fun List<TreeCloneClass>.validClonesFilter(): List<CloneClass> =
    mapNotNull { it.remainValidClones() }

fun TreeCloneClass.remainValidClones(): CloneClass? {
    val clones = clones.filter { it.firstElement.isValid }.toList()
    return if (clones.size > 1) {
        RangeCloneClass(clones)
    } else {
        null
    }
}
