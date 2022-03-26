package org.jetbrains.research.cloneDetector.core.postprocessing

import org.jetbrains.research.cloneDetector.core.SuffixTree
import org.jetbrains.research.cloneDetector.core.getAllCloneClasses
import org.jetbrains.research.cloneDetector.core.structures.Clone
import org.jetbrains.research.cloneDetector.core.structures.CloneClass
import org.jetbrains.research.cloneDetector.core.structures.SourceToken
import org.jetbrains.research.cloneDetector.core.utils.*
import org.jetbrains.research.cloneDetector.ide.configuration.PluginSettings

fun List<CloneClass>.filterSelfCoveredClasses(): List<CloneClass> = filter(::filterPredicate)

data class CloneScore(val selfCoverage: Double, val sameMethodCount: Double, val length: Int)

fun CloneClass.getScore(): CloneScore =
    CloneScore(scoreSelfCoverage(), scoreSameMethod(), clones.first().textLength)

fun filterPredicate(cloneClass: CloneClass): Boolean =
    Application.readAction {
        with(cloneClass.getScore()) {
            selfCoverage <= PluginSettings.coverageSkipFilter / 100f || selfCoverage <= 0.85 && sameMethodCount <= 0.7
        }
    }

private fun CloneClass.scoreSelfCoverage(): Double =
    clones.first().scoreSelfCoverage()

private fun Clone.scoreSelfCoverage(): Double {
    val sequence = tokenSequence().toList()
    val indexMap = sequence.mapIndexed { i, psiElement -> psiElement to i }.toMap()
    val tree = suffixTree(sequence.map(::SourceToken).toList())

    if (tree.haveTooMuchClones(sequence.size)) return 1.0

    val length = tree
        .getAllCloneClasses(10).toList()
        .filterSubClassClones()
        .flatMap { it.clones.toList() }
        .map { IntRange(indexMap[it.firstPsi]!!, indexMap[it.lastPsi]!!) }
        .uniteRanges()
        .sumOf { it.length }
    return length.toDouble() / sequence.size
}

private fun SuffixTree<SourceToken>.haveTooMuchClones(sourceLength: Int) =
    getAllCloneClasses(10).drop(sourceLength * PluginSettings.coverageSkipFilter / 100).firstOrNull() != null

private fun CloneClass.scoreSameMethod(): Double {
    val mostPopularMethodNumber = clones.map { it.firstPsi.method }.groupBy { it }.map { it.value.size }.maxOrNull()
    return (mostPopularMethodNumber!! - 1) / (size - 1).toDouble()
}
