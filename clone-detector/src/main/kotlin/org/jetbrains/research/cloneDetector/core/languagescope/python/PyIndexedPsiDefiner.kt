package org.jetbrains.research.cloneDetector.core.languagescope.python

import com.intellij.psi.PsiElement
import com.intellij.psi.PsiFile
import com.intellij.psi.PsiMethod
import org.jetbrains.research.cloneDetector.core.languagescope.IndexedPsiDefiner
import org.jetbrains.research.cloneDetector.core.languagescope.java.JavaIndexedSequence
import org.jetbrains.research.cloneDetector.core.structures.IndexedSequence

class PyIndexedPsiDefiner : IndexedPsiDefiner {
    override val fileType: String
        get() = "Python"

    override fun isIndexed(psiElement: PsiElement): Boolean {
        val typeString = psiElement.node.elementType.toString()
        return psiElement is PsiFile
    }

    override fun createIndexedSequence(psiElement: PsiElement): IndexedSequence {
//        require(isIndexed(psiElement))
        print("INDEXED PYTHON ")
        println(psiElement.toString())
        return PyIndexedSequence(psiElement)
    }
}

//override fun isIndexed(psiElement: PsiElement): Boolean =
//    psiElement is PsiMethod
//
//override fun createIndexedSequence(psiElement: PsiElement): IndexedSequence {
//    require(isIndexed(psiElement))
//    return JavaIndexedSequence(psiElement as PsiMethod)
