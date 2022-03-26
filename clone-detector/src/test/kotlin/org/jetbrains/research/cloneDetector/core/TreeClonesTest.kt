package org.jetbrains.research.cloneDetector.core

import org.jetbrains.research.cloneDetector.core.postprocessing.filterSubClassClones
import org.jetbrains.research.cloneDetector.core.structures.CloneClass
import org.jetbrains.research.cloneDetector.core.utils.areEqual
import org.jetbrains.research.cloneDetector.core.utils.tokenSequence

class TreeClonesTest : FolderProjectTest("${getResourcesRootPath(::FolderProjectTest)}/sphinx4-java/") {
    private val clones
        get() = CloneIndexer.getAllCloneClasses().filterSubClassClones().toList()

    fun testNotAloneDuplicate() {
        assertTrue(clones.all(::checkCountInvariant))
    }

    fun testSameTokenLengthSequence() {
        assertTrue(clones.all(::checkTokenLengthInvariant))
    }
}

fun checkCountInvariant(cloneClass: CloneClass): Boolean =
    cloneClass.clones.count() > 1

fun checkTokenLengthInvariant(cloneClass: CloneClass): Boolean =
    cloneClass.clones.map { it.tokenSequence().count() }.areEqual()
