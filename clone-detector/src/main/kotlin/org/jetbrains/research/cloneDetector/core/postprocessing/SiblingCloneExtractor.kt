package org.jetbrains.research.cloneDetector.core.postprocessing

import com.intellij.psi.PsiElement
import org.jetbrains.research.cloneDetector.core.postprocessing.helpers.cropBadTokens
import org.jetbrains.research.cloneDetector.core.postprocessing.helpers.extractSiblingSequences
import org.jetbrains.research.cloneDetector.core.structures.*
import org.jetbrains.research.cloneDetector.core.utils.*
import org.jetbrains.research.cloneDetector.ide.configuration.PluginSettings

fun List<CloneClass>.splitSiblingClones(): List<CloneClass> = flatMap(CloneClass::splitToSiblingClones)

private fun CloneClass.splitToSiblingClones(): List<CloneClass> {
    val minTokenLength = PluginSettings.minCloneLength
    normalizePsiHierarchy().run {
        val randomClone = clones.first()
        val siblingClones = randomClone
            .extractSiblingSequences()
            .map(PsiRange::cropBadTokens)
            .filter { it.tokenSequence().count() > minTokenLength }
            .toList()
        val siblingRanges = siblingClones.mapToTokenIndexes(randomClone.tokenSequence())
        return clones.map { it.extractSubClones(siblingRanges).asSequence() }.zipped().map(::RangeCloneClass)
    }
}

private fun PsiElement.getNextGoodElement(): PsiElement {
    var current = this
    while (isNoiseElement(current)) current = current.nextLeafElement()!!
    return current
}

private fun PsiElement.getPrevGoodElement(): PsiElement {
    var current = this
    while (isNoiseElement(current)) current = current.prevLeafElement()
    return current
}

private fun Clone.extractSubClones(intervals: List<IntRange>): List<Clone> {
    val sequence = tokenSequence().toList()
    return intervals.map { range -> RangeClone(sequence[range.start], sequence[range.endInclusive]) }
}

private fun List<Clone>.mapToTokenIndexes(container: Sequence<PsiElement>): List<IntRange> {
    val map = container.mapIndexed { i, psiElement -> psiElement to i }.toMap()
    return this.map {
        IntRange(
            map[it.firstPsi.firstEndChild().getNextGoodElement()]!!,
            map[it.lastPsi.lastEndChild().getPrevGoodElement()]!!
        )
    }
}

/**
 * Finds the biggest parent for firstPsi which points at the same place
 */
private fun Clone.normalizePsiHierarchy(): Clone {
    var current = firstPsi
    while (current.textRange.startOffset == current.parent.textRange.startOffset &&
        current.parent.textRange.endOffset <= lastPsi.textRange.endOffset)
        current = current.parent
    return RangeClone(current, lastPsi)
}

private fun CloneClass.normalizePsiHierarchy(): CloneClass =
    RangeCloneClass(clones.map(Clone::normalizePsiHierarchy).toList())
