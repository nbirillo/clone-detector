package org.jetbrains.research.cloneDetector.core

import com.intellij.openapi.vfs.VirtualFile
import com.intellij.psi.PsiFile
import org.jetbrains.kotlin.idea.core.util.end
import org.jetbrains.kotlin.idea.core.util.start
import org.jetbrains.research.cloneDetector.core.languagescope.java.JavaIndexedPsiDefiner
import org.jetbrains.research.cloneDetector.core.postprocessing.filterSubClassClones
import org.jetbrains.research.cloneDetector.core.structures.SourceToken
import org.jetbrains.research.cloneDetector.core.structures.TreeCloneClass
import org.jetbrains.research.cloneDetector.core.utils.addIf
import org.jetbrains.research.cloneDetector.core.utils.riseTraverser
import org.jetbrains.research.cloneDetector.core.utils.tokenSequence
import org.jetbrains.research.cloneDetector.ide.configuration.PluginSettings
import java.util.*
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write
import com.google.gson.Gson
import com.intellij.psi.util.elementType
import org.jetbrains.research.cloneDetector.core.languagescope.python.PyIndexedPsiDefiner
import org.jetbrains.research.cloneDetector.core.utils.asSequence


data class ClonesDataToJson(
    val clonesGroupsCount: Int,
    val totalClonesCount: Int,
    val clonesCountByGroups: List<Int>,
    val tokenLengthByGroups: List<Int>,
    val clonesPositions: List<List<List<List<Int>>>>?
)

object CloneIndexer {
    internal var tree = SuffixTree<SourceToken>()
    internal val rwLock = ReentrantReadWriteLock()

    var indexedTokens: Long = 0
        private set

    fun clear() {
        fileSequenceIds.clear()
        tree = SuffixTree()
    }

    val fileSequenceIds = HashMap<VirtualFile, List<Long>>()

    fun addFile(psiFile: PsiFile): Unit = rwLock.write {
        if (psiFile.virtualFile in fileSequenceIds) return
//        val indexedPsiDefiner = psiFile.project.languageSerializer.getIndexedPsiDefiner(psiFile)
        print("CLONE INDEXER BEFORE SEQUENCE SIZE ${psiFile.asSequence().toList().size}")
//        println(psiFile.asSequence())
//        val indexedPsiDefiner = JavaIndexedPsiDefiner()
        val indexedPsiDefiner = PyIndexedPsiDefiner()

        val ids = mutableListOf<Long>()
//        listOf(psiFile).map {
         indexedPsiDefiner?.getIndexedChildren(psiFile)?.map {
            println("IN ITERATOR ${it.name}")
            println("SEQUENCE ${indexedPsiDefiner.createIndexedSequence(it)}")
            val sequence = indexedPsiDefiner.createIndexedSequence(it).sequence.toList()
//            val sequence = psiFile.asSequence().toList()
            println("SEQUENCE AS LIST $sequence WITH LENGTH ${sequence.size}")

            if (sequence.size > PluginSettings.minCloneLength) {
                indexedTokens += sequence.size
                val id = tree.addSequence(sequence)
                ids += id
            }
        }
        fileSequenceIds.put(psiFile.virtualFile, ids)
    }

    fun removeFile(virtualFile: VirtualFile): Unit = rwLock.write {
        val ids = fileSequenceIds[virtualFile] ?: return
        ids.forEach {
            indexedTokens -= tree.getSequence(it).size
            tree.removeSequence(it)
        }
        fileSequenceIds.remove(virtualFile)
    }

    fun getAllFileCloneClasses(virtualFile: VirtualFile): List<TreeCloneClass> = rwLock.read {
        val ids = fileSequenceIds[virtualFile] ?: return emptyList()
        return ids
            .flatMap { tree.getAllSequenceClasses(it, PluginSettings.minCloneLength / 2).toList() }
            .filterSubClassClones()
    }

    fun getFileCloneClassesGroupedBySequence(virtualFile: VirtualFile): List<List<TreeCloneClass>> {
        val ids = fileSequenceIds[virtualFile] ?: return emptyList()
        return ids.map { tree.getAllSequenceClasses(it, PluginSettings.minCloneLength / 2).toList() }
    }

    fun getAllCloneClasses(): List<TreeCloneClass> = rwLock.read {
        tree.getAllCloneClasses(PluginSettings.minCloneLength)
    }

//    fun getClones(): List<TreeCloneClass> {
//        val file = myFixture.configureByFile("${getResourcesRootPath(::FolderProjectTest)}/debug/SimpleClass.java")!!
//        CloneIndexer.addFile(file)
//        return CloneIndexer.getAllCloneClasses().filterSubClassClones().toList()
//    }

    fun getClonesSubTreesCount(): Int = this.getAllCloneClasses().filterSubClassClones().toList().size

    private fun getClones() = this.getAllCloneClasses().filterSubClassClones().toList()

    fun getClonesGroupsCount() = getClones().size

    fun getTotalClonesCount() = getClonesCountByGroups().sum()

    fun getClonesCountByGroups() = getClones().map { tree -> tree.clones.toList().size }.toList()

    fun getTokenLengthByGroups() = getClones().map { tree -> tree.clones.first().tokenSequence().toList().size }.toList()

    fun getClonesPositions() = getClones().map{ tree ->
        tree.clones.map {
            it.tokenSequence().map { el ->
                val start = el.textRange.start
                val end = el.textRange.end
                listOf(start, end)
            }.toList()
        }.toList()
    }.toList()


    fun dataToJson(isTokenLength: Boolean): String {
        val gson = Gson()

        return gson.toJson(
            ClonesDataToJson(
                getClonesGroupsCount(),
                getTotalClonesCount(),
                getClonesCountByGroups(),
                getTokenLengthByGroups(),
                if (isTokenLength) getClonesPositions() else null
            )
        )
    }
}

fun Node.visitChildren(visit: (Node) -> Unit) {
    visit(this)
    this.edges.mapNotNull { it.terminal }.forEach { it.visitChildren(visit) }
}

fun SuffixTree<SourceToken>.getAllCloneClasses(minTokenLength: Int): List<TreeCloneClass> {
    val clones = ArrayList<TreeCloneClass>()
    root.visitChildren {
        val cloneClass = TreeCloneClass(it)
        if (cloneClass.length > minTokenLength) {
            clones.add(cloneClass)
        }
    }
    return clones
}

fun SuffixTree<SourceToken>.getAllSequenceClasses(id: Long, minTokenLength: Int): Sequence<TreeCloneClass> {
    val classes = LinkedList<TreeCloneClass>()
    val visitedNodes = HashSet<Node>()
    for (branchNode in this.getAllLastSequenceNodes(id)) {
        for (currentNode in branchNode.riseTraverser()) {
            if (visitedNodes.contains(currentNode)) break
            visitedNodes.add(currentNode)
            classes.addIf(TreeCloneClass(currentNode)) { it.length > minTokenLength }
        }
    }
    return classes.asSequence()
}
