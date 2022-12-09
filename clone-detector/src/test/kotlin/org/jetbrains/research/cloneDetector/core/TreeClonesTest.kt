package org.jetbrains.research.cloneDetector.core

import com.google.gson.Gson
import com.google.gson.JsonObject
import org.jetbrains.research.cloneDetector.core.postprocessing.filterSubClassClones
import org.jetbrains.research.cloneDetector.core.postprocessing.helpers.cropBadTokens
import org.jetbrains.research.cloneDetector.core.postprocessing.helpers.extractSiblingSequences
import org.jetbrains.research.cloneDetector.core.postprocessing.validClonesFilter
import org.jetbrains.research.cloneDetector.core.structures.CloneClass
import org.jetbrains.research.cloneDetector.core.structures.PsiRange
import org.jetbrains.research.cloneDetector.core.structures.TreeClone
import org.jetbrains.research.cloneDetector.core.structures.TreeCloneClass
import org.jetbrains.research.cloneDetector.core.utils.areEqual
import org.jetbrains.research.cloneDetector.core.utils.printText
import org.jetbrains.research.cloneDetector.core.utils.tokenSequence
import org.jetbrains.research.cloneDetector.ide.configuration.PluginSettings
import java.io.File

//"${getResourcesRootPath(::FolderProjectTest)}/debug/"
class TreeClonesTest : FolderProjectTest("${getResourcesRootPath(::FolderProjectTest)}/debug/") {

    fun getClones(filename: String): List<TreeCloneClass> {

        val file = myFixture.configureByFile("${getResourcesRootPath(::FolderProjectTest)}/debug/$filename")!!
        CloneIndexer.addFile(file)

        return CloneIndexer.getAllCloneClasses().filterSubClassClones().toList()
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
//            it.tokenSequence().forEach { el ->
//                print(el.text)
//            }
            it.printText()
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
        val filename = file.toString().substring(file.toString().lastIndexOf("/") + 1);
        println(filename)
        for (length in 20..20 step 5) {
            PluginSettings.minCloneLength = length

            val clones = getClones(filename)
            println(clones.size)

            clones.forEachIndexed { i, it ->
                print("($i.) ")
                for (clone in it.clones) {
//                    println("${clone.startLine}, ${clone.endLine}")
                    clone.printText()
                    println("+++++++++++++++")
                }
//                it.printText()
                println("\n----------------------")
            }
//            getClonesText(clones.asSequence())
//            clones.forEach{ tree ->
//                tree.clones.forEach {
//                    // (${i.text}
//                    println(it.tokenSequence().toList().size)
//                    it.tokenSequence().toList().forEach { i -> print("${i.node.elementType}(${i.text}) -> ") }
//                    println()
//                    println("-------------------")
//                }
//                println("=====================")
//            }

//            println(CloneIndexer.getClonesGroupsCount())
//            println(CloneIndexer.getClonesCountByGroups())
            println(CloneIndexer.getTokenLengthByGroups())
//            println(CloneIndexer.getTotalClonesCount())

            jsonObject.addProperty(
                "${PluginSettings.minCloneLength}",
                CloneIndexer.dataToJson(isTokenLength = false)
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
        PluginSettings.minCloneLength = 3

        File("${getResourcesRootPath(::FolderProjectTest)}/debug/").walkTopDown().forEach { it ->
            if (it.extension == "py" && it.nameWithoutExtension == "notebook8075") {
                val gson = Gson()
                val jsonObject = JsonObject()

                val filename = it.toString().substring(it.toString().lastIndexOf("/") + 1)
                val clones = getClones(filename)

                jsonObject.addProperty(
                    "${PluginSettings.minCloneLength}",
                    CloneIndexer.dataToJson2(isTokenLength = false)
                )

                val targetDir = "/Users/konstantingrotov/Documents/programming/data/clones/test/"
                val scriptFile = File("${targetDir}${filename}.json")
                val file = myFixture.configureByFile("${getResourcesRootPath(::FolderProjectTest)}/debug/$filename")!!

                scriptFile.writeText(Gson().toJson(jsonObject).toString())
                CloneIndexer.removeFile(file.virtualFile)

            }
        }
    }
}

fun checkCountInvariant(cloneClass: CloneClass): Boolean =
    cloneClass.clones.count() > 1

fun checkTokenLengthInvariant(cloneClass: CloneClass): Boolean =
    cloneClass.clones.map { it.tokenSequence().count() }.areEqual()
