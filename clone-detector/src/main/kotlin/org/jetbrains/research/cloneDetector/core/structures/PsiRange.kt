package org.jetbrains.research.cloneDetector.core.structures

import com.intellij.psi.PsiElement

interface PsiRange {
    val firstPsi: PsiElement
    val lastPsi: PsiElement
}
