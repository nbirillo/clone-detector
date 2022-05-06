package org.jetbrains.research.cloneDetector.core

import com.google.gson.Gson
import com.google.gson.JsonObject
import org.jetbrains.kotlin.idea.core.util.end
import org.jetbrains.kotlin.idea.core.util.start
import org.jetbrains.research.cloneDetector.core.postprocessing.filterSubClassClones
import org.jetbrains.research.cloneDetector.core.structures.CloneClass
import org.jetbrains.research.cloneDetector.core.structures.TreeClone
import org.jetbrains.research.cloneDetector.core.structures.TreeCloneClass
import org.jetbrains.research.cloneDetector.core.utils.areEqual
import org.jetbrains.research.cloneDetector.core.utils.tokenSequence
import org.jetbrains.research.cloneDetector.ide.configuration.PluginSettings
import java.io.File
import java.io.InputStream

//"${getResourcesRootPath(::FolderProjectTest)}/debug/"
class TreeClonesTest : FolderProjectTest("${getResourcesRootPath(::FolderProjectTest)}/debug/") {

    fun getClones(filename: String): List<TreeCloneClass> {

        val file = myFixture.configureByFile("${getResourcesRootPath(::FolderProjectTest)}/debug/$filename")!!
        CloneIndexer.addFile(file)
        val clones = CloneIndexer.getAllCloneClasses().filterSubClassClones().toList()

        return clones
    }

    fun getClonesText(clones: Sequence<TreeClone>): String {
//        val name = "SimpleFile2.py"
//        val inputStream: InputStream = File("${getResourcesRootPath(::FolderProjectTest)}/debug/$name").inputStream()
//        val fileString = inputStream.bufferedReader().use { it.readText() }
//
//        clones.forEachIndexed { i, it ->
//            print("($i.) ")
//            it.tokenSequence().forEach { el ->
//                val start = el.textRange.start
//                val end = el.textRange.end
//                print(fileString.subSequence(start, end))
//            }
//            println("\n----------------------")
//        }
        clones.forEachIndexed { i, it ->
            print("($i.) ")
            it.tokenSequence().forEach { el ->
                print(el.text)
            }
            println("\n----------------------")
        }
        return ""
    }

    fun getClonesSizeFromObject(): Int {
        val name = "SimpleFile4.py"
        val file = myFixture.configureByFile("${getResourcesRootPath(::FolderProjectTest)}/debug/$name")!!
        CloneIndexer.addFile(file)

        return CloneIndexer.getClonesSubTreesCount()
    }

//    fun testNotAloneDuplicate() {
//
//        var clones = getClones()
//
//        println(clones.size)
//        require(clones.isNotEmpty())
//
//        clones.forEach{ tree ->
//            tree.clones.forEach {
//                println(it.tokenSequence().toList())
//            }
//        }
//        println("========================")
//        assertTrue(clones.all(::checkCountInvariant))
//    }

    fun doAnalysis(file: File) {
        val gson = Gson()
        val jsonObject = JsonObject()
        val filename = file.toString().substring(file.toString().lastIndexOf("/")+1);
        println(filename)
        for (length in 5..10 step 5) {
            PluginSettings.minCloneLength = length

            val clones = getClones(filename)
//            println(clones.size)
//
//            println(CloneIndexer.getClonesGroupsCount())
//            println(CloneIndexer.getClonesCountByGroups())
//            println(CloneIndexer.getTokenLengthByGroups())
//            println(CloneIndexer.getTotalClonesCount())

            jsonObject.addProperty(
                "${PluginSettings.minCloneLength}",
                CloneIndexer.dataToJson(isTokenLength=false)
            )
            assertTrue(clones.all(::checkCountInvariant))
            val filee = myFixture.configureByFile("${getResourcesRootPath(::FolderProjectTest)}/debug/$filename")!!
            CloneIndexer.removeFile(filee.virtualFile)
        }


        val targetDir = "/Users/konstantingrotov/Documents/Programming/tools/clones-analysis/clones_results/notebooks/"
        val scriptFile = File("$targetDir$filename.json")
//        CloneIndexer.removeFile()
        scriptFile.writeText(Gson().toJson(jsonObject).toString())

    }

    fun testClonesSize() {
//        val gson = Gson()
//        val jsonObject = JsonObject()

        File("${getResourcesRootPath(::FolderProjectTest)}/debug/").walkTopDown().forEachIndexed { i, it ->
            if (it.extension == "py") {
                println("($i)")

                doAnalysis(it)
            }
        }

        assertTrue(true)
//        for (length in 5..80 step 5) {
//
//            PluginSettings.minCloneLength = length
//            println(PluginSettings.minCloneLength)
//
//            val clones = getClones()
//
////            print("SIZE ")
////            println(clones.size)
////            clones.forEach { tree ->
////                tree.clones.forEach {
////                    it.tokenSequence().forEach { el ->
////                        print(el.text)
////                    }
////                    println()
////                    println("------------------")
////                }
////                println("====================")
////            }
//
////
//            println(CloneIndexer.getClonesGroupsCount())
//            println(CloneIndexer.getClonesCountByGroups())
//            println(CloneIndexer.getTokenLengthByGroups())
//            println(CloneIndexer.getTotalClonesCount())
//
//            jsonObject.addProperty(
//                "${PluginSettings.minCloneLength}",
//                CloneIndexer.dataToJson(isTokenLength=false)
//            )
//            assertTrue(clones.all(::checkCountInvariant))
//        }
//
//        val targetDir = "/Users/konstantingrotov/Documents/Programming/tools/clones-analysis/data/"
//        val name = "4.json"
//        val scriptFile = File("$targetDir$name")
//        scriptFile.writeText(Gson().toJson(jsonObject).toString())

    }

//    fun testSameTokenLengthSequence() {
//        assertTrue(clones.all(::checkTokenLengthInvariant))
//    }
}

fun checkCountInvariant(cloneClass: CloneClass): Boolean =
    cloneClass.clones.count() > 1

fun checkTokenLengthInvariant(cloneClass: CloneClass): Boolean =
    cloneClass.clones.map { it.tokenSequence().count() }.areEqual()
