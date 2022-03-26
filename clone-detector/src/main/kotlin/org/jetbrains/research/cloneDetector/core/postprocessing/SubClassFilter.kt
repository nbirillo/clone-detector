package org.jetbrains.research.cloneDetector.core.postprocessing

import org.jetbrains.research.cloneDetector.core.Node
import org.jetbrains.research.cloneDetector.core.structures.TreeCloneClass
import java.util.*

fun List<TreeCloneClass>.filterSubClassClones(): List<TreeCloneClass> {
    val subClassFilter = SubclassFilter(this)
    return this.filter { subClassFilter.isAllowed(it) }
}

/*
 * Consider example for given duplicates {a,b,c,d,e} & {a,b,c,d,e}
 * There are other clones found by suffix tree: {c,d,e} & {c,d,e}, {c,d,e} & {c,d,e}, {d,e} & {d,e} which are redundant
 * In this case suffix link connects to the smaller class
 *
 */
private class SubclassFilter(treeCloneClassesToFilter: Iterable<TreeCloneClass>) {

    /*
     * link from node to suffixTreeCloneClass with suffix link to this node
     */
    private val reverseSuffixLink by lazy {
        IdentityHashMap<Node, TreeCloneClass>()
            .apply {
                treeCloneClassesToFilter.asSequence().filter { it.treeNode.suffixLink != null }
                    .forEach { put(it.treeNode.suffixLink, it) }
            }
    }

    fun isAllowed(treeCloneClass: TreeCloneClass): Boolean {
        val greaterClass = reverseSuffixLink[treeCloneClass.treeNode] ?: return true
        return greaterClass.size != treeCloneClass.size
    }
}
