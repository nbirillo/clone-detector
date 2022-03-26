package org.jetbrains.research.cloneDetector.core

import org.junit.Test


class SuffixTreeRemovingTest {
    private var tree = SuffixTree<CharToken>()
    private var sequence1 = "cacao".map(::CharToken)
    private var sequence2 = "cacaoa".map(::CharToken)

    @Test
    fun testSimpleRemoveSequence() {
        println("Test simple remove sequence")
        tree.addSequence(sequence1)
        val id2 = tree.addSequence(sequence2)
        println(tree)
        tree.removeSequence(id2)
        println(tree)
    }
}
