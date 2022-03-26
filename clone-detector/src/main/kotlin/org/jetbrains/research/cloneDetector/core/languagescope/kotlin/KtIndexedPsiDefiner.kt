package org.jetbrains.research.cloneDetector.core.languagescope.kotlin

import com.intellij.psi.PsiElement
import org.jetbrains.research.cloneDetector.core.languagescope.IndexedPsiDefiner
import org.jetbrains.research.cloneDetector.core.structures.IndexedSequence

class KtIndexedPsiDefiner : IndexedPsiDefiner {
    override val fileType: String
        get() = "Kotlin"

    override fun isIndexed(psiElement: PsiElement): Boolean {
        val typeString = psiElement.node.elementType.toString()
        return typeString == "FUN" || typeString == "PROPERTY_ACCESSOR"
    }

    override fun createIndexedSequence(psiElement: PsiElement): IndexedSequence {
        require(isIndexed(psiElement))
        return KtIndexedSequence(psiElement)
    }
}
