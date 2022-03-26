package org.jetbrains.research.cloneDetector.core.languagescope

import com.intellij.openapi.fileTypes.FileType
import com.intellij.psi.PsiElement
import org.jetbrains.research.cloneDetector.core.structures.IndexedSequence
import org.jetbrains.research.cloneDetector.core.utils.leafTraverse

/**
 * Defines files to be indexed
 * Provides file type, PSI Element, possible parents of the Element witch must be indexed
 */
interface IndexedPsiDefiner {
    /**
     * Defines files to be indexed
     * @see FileType.getName
     */
    val fileType: String

    fun createIndexedSequence(psiElement: PsiElement): IndexedSequence

    /**
     * Defines elements to be indexed
     * @see isIndexed
     */
    fun isIndexed(psiElement: PsiElement): Boolean

    fun getIndexedChildren(psiElement: PsiElement): List<PsiElement> =
        psiElement.leafTraverse({ isIndexed(it) }) { it.children.asSequence() }.toList()
}
