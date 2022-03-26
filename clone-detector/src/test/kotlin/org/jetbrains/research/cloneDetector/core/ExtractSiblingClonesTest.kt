package org.jetbrains.research.cloneDetector.core

import org.jetbrains.research.cloneDetector.core.postprocessing.filterSubClassClones
import org.jetbrains.research.cloneDetector.core.postprocessing.mergeCloneClasses
import org.jetbrains.research.cloneDetector.core.postprocessing.splitSiblingClones
import org.jetbrains.research.cloneDetector.core.structures.CloneClass
import org.jetbrains.research.cloneDetector.core.utils.printText
import org.jetbrains.research.cloneDetector.core.utils.tokenSequence


class ExtractSiblingClonesTest :
    FolderProjectTest("${getResourcesRootPath(::FolderProjectTest)}/sphinx4-java/") {
    private val clones
        get() = CloneIndexer.getAllCloneClasses().filterSubClassClones()

    fun testNotAloneDuplicate() {
        val problems = clones.splitSiblingClones().mergeCloneClasses().filter { !checkCountInvariant(it) }
        problems.forEach(CloneClass::printInfo)
        assertTrue(problems.isEmpty())
    }

    fun testSameTokenLengthSequence() {
        val problems = clones.splitSiblingClones().mergeCloneClasses().filter { !checkTokenLengthInvariant(it) }
        problems.forEach {
            it.printInfo()
            it.clones.forEach {
                println(it.tokenSequence().toList())
            }
        }
        assertTrue(problems.isEmpty())
    }
}

fun CloneClass.printInfo() {
    clones.forEach {
        println("========================")
        println("Problem class:")
        println(it.tokenSequence().toList())
        it.printText()
        println("------------------------")
    }
}
