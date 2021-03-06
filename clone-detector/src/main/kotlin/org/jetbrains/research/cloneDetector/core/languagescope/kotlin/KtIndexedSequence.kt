package org.jetbrains.research.cloneDetector.core.languagescope.kotlin

import com.intellij.psi.PsiElement
import org.jetbrains.research.cloneDetector.core.structures.IndexedSequence
import org.jetbrains.research.cloneDetector.core.structures.SourceToken
import org.jetbrains.research.cloneDetector.core.utils.depthFirstTraverse
import org.jetbrains.research.cloneDetector.core.utils.isNoiseElement

class KtIndexedSequence(val psiElement: PsiElement) : IndexedSequence {
    override val sequence: Sequence<SourceToken>
        get() = psiElement.toSequence().map(::SourceToken)
}

private fun PsiElement.toSequence(): Sequence<PsiElement> =
    lastChild.depthFirstTraverse { it.psiChildren }.filter { it.firstChild == null }.filterNot(::isNoiseElement)

/**
 * PsiElement.children returns only KtElements
 * This property returns all psi children
 */
private val PsiElement.psiChildren: Sequence<PsiElement>
    get() = generateSequence(this.firstChild) { it.nextSibling }
