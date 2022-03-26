package org.jetbrains.research.cloneDetector.core.languagescope.java

import com.intellij.psi.PsiElement
import com.intellij.psi.PsiMethod
import org.jetbrains.research.cloneDetector.core.languagescope.IndexedPsiDefiner
import org.jetbrains.research.cloneDetector.core.structures.IndexedSequence

class JavaIndexedPsiDefiner : IndexedPsiDefiner {
    override val fileType: String
        get() = "JAVA"

    override fun isIndexed(psiElement: PsiElement): Boolean =
        psiElement is PsiMethod

    override fun createIndexedSequence(psiElement: PsiElement): IndexedSequence {
        require(isIndexed(psiElement))
        return JavaIndexedSequence(psiElement as PsiMethod)
    }
}
