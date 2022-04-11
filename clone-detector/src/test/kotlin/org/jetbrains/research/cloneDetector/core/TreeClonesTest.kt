package org.jetbrains.research.cloneDetector.core

import org.jetbrains.research.cloneDetector.core.postprocessing.filterSubClassClones
import org.jetbrains.research.cloneDetector.core.structures.CloneClass
import org.jetbrains.research.cloneDetector.core.structures.TreeCloneClass
import org.jetbrains.research.cloneDetector.core.utils.areEqual
import org.jetbrains.research.cloneDetector.core.utils.printText
import org.jetbrains.research.cloneDetector.core.utils.tokenSequence
import org.junit.Before


class TreeClonesTest : FolderProjectTest("${getResourcesRootPath(::FolderProjectTest)}/debug/") {


    fun getClones(): List<TreeCloneClass> {
        val file = myFixture.configureByFile("${getResourcesRootPath(::FolderProjectTest)}/debug/SimpleClass.java")!!
        CloneIndexer.addFile(file)
        return CloneIndexer.getAllCloneClasses().filterSubClassClones().toList()
    }

//    private val clones
//        get() = CloneIndexer.getAllCloneClasses().filterSubClassClones().toList()

    fun testNotAloneDuplicate() {

        var clones = getClones()

        println(clones.size)
        require(clones.isNotEmpty())

        clones.forEach{ tree ->
            tree.clones.forEach { 
                println(it.tokenSequence().toList())
            }
        }
        println("========================")
        assertTrue(clones.all(::checkCountInvariant))
    }

//    fun testSameTokenLengthSequence() {
//        assertTrue(clones.all(::checkTokenLengthInvariant))
//    }
}

fun checkCountInvariant(cloneClass: CloneClass): Boolean =
    cloneClass.clones.count() > 1

fun checkTokenLengthInvariant(cloneClass: CloneClass): Boolean =
    cloneClass.clones.map { it.tokenSequence().count() }.areEqual()
