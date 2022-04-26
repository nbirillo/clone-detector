//package org.jetbrains.research.cloneDetector.core
//
//import com.google.gson.Gson
//import com.google.gson.JsonObject
//import org.jetbrains.kotlin.idea.core.util.end
//import org.jetbrains.kotlin.idea.core.util.start
//import org.jetbrains.research.cloneDetector.core.postprocessing.filterSubClassClones
//import org.jetbrains.research.cloneDetector.core.structures.CloneClass
//import org.jetbrains.research.cloneDetector.core.structures.TreeClone
//import org.jetbrains.research.cloneDetector.core.structures.TreeCloneClass
//import org.jetbrains.research.cloneDetector.core.utils.areEqual
//import org.jetbrains.research.cloneDetector.core.utils.tokenSequence
//import org.jetbrains.research.cloneDetector.ide.configuration.PluginSettings
//import java.io.File
//import java.io.InputStream
//
//
//fun getClones(file: File): List<TreeCloneClass> {
//    CloneIndexer.addFile(file)
//    return CloneIndexer.getAllCloneClasses().filterSubClassClones().toList()
//}
//
//fun getClonesText(clones: Sequence<TreeClone>): String {
//    clones.forEachIndexed { i, it ->
//        print("($i.) ")
//        it.tokenSequence().forEach { el ->
//            print(el.text)
//        }
//        println("\n----------------------")
//    }
//    return ""
//}
//
//fun doAnalysis(file: File) {
//    val gson = Gson()
//    val jsonObject = JsonObject()
//
//    for (length in 80..80 step 5) {
//        PluginSettings.minCloneLength = length
//
//        println(file.toString())
//        val clones = getClones(file.toString())
////            println(clones.size)
////
////            println(CloneIndexer.getClonesGroupsCount())
////            println(CloneIndexer.getClonesCountByGroups())
////            println(CloneIndexer.getTokenLengthByGroups())
////            println(CloneIndexer.getTotalClonesCount())
//
//        jsonObject.addProperty(
//            "${PluginSettings.minCloneLength}",
//            CloneIndexer.dataToJson(isTokenLength=false)
//        )
//    }
//
//    val targetDir = "/Users/konstantingrotov/Documents/Programming/tools/clones-analysis/clones_results/notebooks/"
//    val name = file.toString().substring(file.toString().lastIndexOf("/")+1);
//    val scriptFile = File("$targetDir$name.json")
////        CloneIndexer.removeFile()
//    scriptFile.writeText(Gson().toJson(jsonObject).toString())
//
//}
//
//fun testClonesSize() {
////        val gson = Gson()
////        val jsonObject = JsonObject()
//
//    File("/Users/konstantingrotov/Documents/Programming/datasets/Lupa-duplicates/data/notebooks").walkTopDown().forEachIndexed { i, it ->
//        if (it.extension == "py") {
//            println("($i)")
//            doAnalysis(it)
//        }
//    }
//
//    assertTrue(true)
////        for (length in 5..80 step 5) {
////
////            PluginSettings.minCloneLength = length
////            println(PluginSettings.minCloneLength)
////
////            val clones = getClones()
////
//////            print("SIZE ")
//////            println(clones.size)
//////            clones.forEach { tree ->
//////                tree.clones.forEach {
//////                    it.tokenSequence().forEach { el ->
//////                        print(el.text)
//////                    }
//////                    println()
//////                    println("------------------")
//////                }
//////                println("====================")
//////            }
////
//////
////            println(CloneIndexer.getClonesGroupsCount())
////            println(CloneIndexer.getClonesCountByGroups())
////            println(CloneIndexer.getTokenLengthByGroups())
////            println(CloneIndexer.getTotalClonesCount())
////
////            jsonObject.addProperty(
////                "${PluginSettings.minCloneLength}",
////                CloneIndexer.dataToJson(isTokenLength=false)
////            )
////            assertTrue(clones.all(::checkCountInvariant))
////        }
////
////        val targetDir = "/Users/konstantingrotov/Documents/Programming/tools/clones-analysis/data/"
////        val name = "4.json"
////        val scriptFile = File("$targetDir$name")
////        scriptFile.writeText(Gson().toJson(jsonObject).toString())
//
//}
//
////    fun testSameTokenLengthSequence() {
////        assertTrue(clones.all(::checkTokenLengthInvariant))
////    }
