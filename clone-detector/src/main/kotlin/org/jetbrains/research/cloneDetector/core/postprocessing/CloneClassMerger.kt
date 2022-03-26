package org.jetbrains.research.cloneDetector.core.postprocessing

import com.intellij.openapi.vfs.VirtualFile
import org.jetbrains.research.cloneDetector.core.UnionFindSet
import org.jetbrains.research.cloneDetector.core.structures.Clone
import org.jetbrains.research.cloneDetector.core.structures.CloneClass
import org.jetbrains.research.cloneDetector.core.structures.RangeCloneClass
import org.jetbrains.research.cloneDetector.core.utils.enshure
import org.jetbrains.research.cloneDetector.core.utils.file

fun List<CloneClass>.mergeCloneClasses(): List<CloneClass> {
    enshure { all { it.clones.take(2).count() > 1 } }
    val unionSet = UnionFindSet(this.flatMap { it.clones.toList() }.map(::CloneID))
    this.forEach {
        val first = CloneID(it.clones.first())
        it.clones.forEach { unionSet.join(first, CloneID(it)) }
    }
    return unionSet.equivalenceClasses.map { RangeCloneClass(it.map { it.clone }) }
}

private class CloneID(val clone: Clone) {
    val virtualFile: VirtualFile
        get() {
            return clone.file
        }

    val left: Int
        get() = clone.firstPsi.textRange.startOffset

    val right: Int
        get() = clone.lastPsi.textRange.endOffset

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other?.javaClass != javaClass) return false

        other as CloneID

        if (virtualFile != other.virtualFile) return false
        if (left != other.left) return false
        if (right != other.right) return false

        return true
    }

    override fun hashCode(): Int {
        var result = left.hashCode()
        result = 31 * result + right.hashCode()
        result = 31 * result + virtualFile.hashCode()
        return result
    }
}
