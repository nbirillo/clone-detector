package org.jetbrains.research.cloneDetector.core.postprocessing

import org.jetbrains.research.cloneDetector.core.structures.TreeCloneClass

fun List<TreeCloneClass>.notLongestSequenceFilter(): List<TreeCloneClass> =
    filter { it.treeNode.edges.all { it.terminal == null } }
