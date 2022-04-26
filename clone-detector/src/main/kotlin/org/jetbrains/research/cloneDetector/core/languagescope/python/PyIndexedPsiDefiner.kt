package org.jetbrains.research.cloneDetector.core.languagescope.python

import com.intellij.psi.PsiElement
import com.intellij.psi.PsiFile
import org.jetbrains.research.cloneDetector.core.languagescope.IndexedPsiDefiner
import org.jetbrains.research.cloneDetector.core.structures.IndexedSequence

class PyIndexedPsiDefiner : IndexedPsiDefiner {
    override val fileType: String
        get() = "Python"

    override fun isIndexed(psiElement: PsiElement): Boolean {
        val typeString = psiElement.node.elementType.toString()
        return psiElement is PsiFile
    }

    override fun createIndexedSequence(psiElement: PsiElement): IndexedSequence {
        require(isIndexed(psiElement))
        return PyIndexedSequence(psiElement)
    }
}
