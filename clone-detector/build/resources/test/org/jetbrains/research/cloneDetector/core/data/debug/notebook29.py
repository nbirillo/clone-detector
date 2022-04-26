#!/usr/bin/env python
# coding: utf-8

# # Ripple Carry Adder Kata
# 
# The **Ripple Carry Adder** quantum kata is a series of exercises designed
# to get you familiar with ripple carry addition on a quantum computer.
# 
# * The simplest quantum adder, covered in part I, closely mirrors its classical counterpart,
# using the same basic components and the same algorithm.
# * Part II explores building an in-place adder.
# * A more complex version of an in-place adder covered in part III of the kata uses a different algorithm
# to reduce the number of ancillary qubits needed.
# * Finally, part IV covers building an in-place quantum subtractor.
# 
# It is recommended to complete the [BasicGates kata](./../BasicGates/BasicGates.ipynb) before this one to get familiar with the basic gates used in quantum computing. The list of basic gates available in Q# can be found at [Microsoft.Quantum.Intrinsic](https://docs.microsoft.com/qsharp/api/qsharp/microsoft.quantum.intrinsic). For the syntax of flow control statements in Q#, see [the Q# documentation](https://docs.microsoft.com/quantum/language/statements#control-flow).
# 
# Each task is wrapped in one operation preceded by the description of the task.
# Your goal is to fill in the blank (marked with // ... comments)
# with some Q# code that solves the task. To verify your answer, run the cell using Ctrl/⌘+Enter.
# 
# Within each section, tasks are given in approximate order of increasing difficulty; harder ones are marked with asterisks.

# To begin, first prepare this notebook for execution (if you skip the first step, you'll get "Syntax does not match any known patterns" error when you try to execute Q# code in the next cells; if you skip the second step, you'll get "Invalid kata name" error):

# In[ ]:


get_ipython().run_line_magic('package', 'Microsoft.Quantum.Katas::0.10.1911.1607')


# > The package versions in the output of the cell above should always match. If you are running the Notebooks locally and the versions do not match, please install the IQ# version that matches the version of the `Microsoft.Quantum.Katas` package.
# > <details>
# > <summary><u>How to install the right IQ# version</u></summary>
# > For example, if the version of `Microsoft.Quantum.Katas` package above is 0.1.2.3, the installation steps are as follows:
# >
# > 1. Stop the kernel.
# > 2. Uninstall the existing version of IQ#:
# >        dotnet tool uninstall microsoft.quantum.iqsharp -g
# > 3. Install the matching version:
# >        dotnet tool install microsoft.quantum.iqsharp -g --version 0.1.2.3
# > 4. Reinstall the kernel:
# >        dotnet iqsharp install
# > 5. Restart the Notebook.
# > </details>

# In[ ]:


get_ipython().run_line_magic('workspace', 'reload')


# ## Part I. Simple Adder Outputting to Empty Qubits
# 
# 
# ### Theory
# 
# * [Classical binary adder on Wikipedia](https://en.wikipedia.org/wiki/Adder_(electronics)).
# * Part 2 of the [paper on quantum binary addition](https://arxiv.org/pdf/quant-ph/0008033.pdf) by Thomas G. Draper explains how to adapt the classical adder to a quantum environment.

# ### Task 1.1. Summation of two bits
# 
# **Inputs:**
# 
#   1. qubit `a` in an arbitrary state $|\phi\rangle$,  
#   2. qubit `b` in an arbitrary state $|\psi\rangle$,  
#   3. qubit `sum` in state $|0\rangle$.
# 
# **Goal:** Transform the `sum` qubit into the lowest bit of the binary sum of $\phi$ and $\psi$.
# 
# * $|0\rangle + |0\rangle \to |0\rangle$
# * $|0\rangle + |1\rangle \to |1\rangle$
# * $|1\rangle + |0\rangle \to |1\rangle$
# * $|1\rangle + |1\rangle \to |0\rangle$
# 
# Any superposition should map appropriately. 
# 
# **Example:** (Recall that $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$, $|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$)
# 
# $|+\rangle \otimes |-\rangle \otimes |0\rangle \to \frac{1}{2}(|000\rangle + |101\rangle - |011\rangle - |110\rangle)$

# In[ ]:


get_ipython().run_line_magic('kata', 'T11_LowestBitSum_Test')

operation LowestBitSum (a : Qubit, b : Qubit, sum : Qubit) : Unit is Adj {
    // ...
}


# ### Task 1.2. Carry of two bits
# 
# **Inputs:**
# 
#   1. qubit `a` in an arbitrary state $|\phi\rangle$,
#   2. qubit `b` in an arbitrary state $|\psi\rangle$,
#   3. qubit `carry` in state $|0\rangle$.
# 
# **Goal:** Set the `carry` qubit to $|1\rangle$ if the binary sum of $\phi$ and $\psi$ produces a carry.
# 
# * $|0\rangle$ and $|0\rangle \to |0\rangle$
# * $|0\rangle$ and $|1\rangle \to |0\rangle$
# * $|1\rangle$ and $|0\rangle \to |0\rangle$
# * $|1\rangle$ and $|1\rangle \to |1\rangle$
# 
# Any superposition should map appropriately. 
# 
# **Example:**
# 
# $|+\rangle \otimes |-\rangle \otimes |0\rangle \to \frac{1}{2}(|000\rangle + |100\rangle - |010\rangle - |111\rangle)$

# In[ ]:


get_ipython().run_line_magic('kata', 'T12_LowestBitCarry_Test')

operation LowestBitCarry (a : Qubit, b : Qubit, carry : Qubit) : Unit is Adj {
    // ...
}


# ### Task 1.3. One-bit adder
# 
# **Inputs:**
# 
#   1. qubit `a` in an arbitrary state $|\phi\rangle$,
#   2. qubit `b` in an arbitrary state $|\psi\rangle$,
#   3. two qubits `sum` and `carry` in state $|0\rangle$.
# 
# **Goals:**
# 
# * Transform the `sum` qubit into the lowest bit of the binary sum of $\phi$ and $\psi$.
# * Transform the `carry` qubit into the carry bit produced by that sum.

# In[ ]:


get_ipython().run_line_magic('kata', 'T13_OneBitAdder_Test')

operation OneBitAdder (a : Qubit, b : Qubit, sum : Qubit, carry : Qubit) : Unit is Adj {
    // ...
}


# ### Task 1.4. Summation of 3 bits
# 
# **Inputs:**
# 
#   1. qubit `a` in an arbitrary state $|\phi\rangle$,
#   2. qubit `b` in an arbitrary state $|\psi\rangle$,
#   3. qubit `carryin` in an arbitrary state $|\omega\rangle$,
#   4. qubit `carryout` in state $|0\rangle$.
# 
# **Goal:** Transform the `sum` qubit into the lowest bit of the binary sum of $\phi$, $\psi$ and $\omega$.

# In[ ]:


get_ipython().run_line_magic('kata', 'T14_HighBitSum_Test')

operation HighBitSum (a : Qubit, b : Qubit, carryin : Qubit, sum : Qubit) : Unit is Adj {
    // ...
}


# ### Task 1.5. Carry of 3 bits
# 
# **Inputs:**
# 
#   1. qubit `a` in an arbitrary state $|\phi\rangle$,
#   2. qubit `b` in an arbitrary state $|\psi\rangle$,
#   3. qubit `carryin` in an arbitrary state $|\omega\rangle$,
#   4. qubit `carryout` in state $|0\rangle$.
# 
# **Goal:** Transform the `carryout` qubit into the carry bit produced by the sum of $\phi$, $\psi$ and $\omega$.

# In[ ]:


get_ipython().run_line_magic('kata', 'T15_HighBitCarry_Test')

operation HighBitCarry (a : Qubit, b : Qubit, carryin : Qubit, carryout : Qubit) : Unit is Adj {
    // ...
}


# ### Task 1.6. Two-bit adder
# 
# **Inputs:**
# 
#   1. two-qubit register `a` in an arbitrary state $|\phi\rangle$,
#   2. two-qubit register `b` in an arbitrary state $|\psi\rangle$,
#   3. two-qubit register `sum` in state $|00\rangle$,
#   4. qubit `carry` in state $|0\rangle$.
# 
# **Goals:**
# 
# * Transform the `sum` register into the binary sum (little-endian) of $\phi$ and $\psi$.
# * Transform the `carry` qubit into the carry bit produced by that sum.
# 
# > All registers in this kata are in **little-endian** order.
# > This means that they have the least significant bit first, followed by the next least significant, and so on.
# >
# > In this exercise, for example, $1$ would be represented as $|10\rangle$, and $2$ as $|01\rangle$.
# >
# > The sum of $|10\rangle$ and $|11\rangle$ would be $|001\rangle$, with the last qubit being the carry qubit.
# 
# <br/>
# <details>
#     <summary>Need a hint? Click here</summary>
#     Don't forget that you can allocate extra qubits.
# </details>

# In[ ]:


get_ipython().run_line_magic('kata', 'T16_TwoBitAdder_Test')

operation TwoBitAdder (a : Qubit[], b : Qubit[], sum : Qubit[], carry : Qubit) : Unit is Adj {
    // ...
}


# ### Task 1.7. N-bit adder
# 
# **Inputs:**
# 
#   1. $N$-qubit register `a` in an arbitrary state $|\phi\rangle$,
#   2. $N$-qubit register `b` in an arbitrary state $|\psi\rangle$,
#   3. $N$-qubit register `sum` in state $|0...0\rangle$,
#   4. qubit `carry` in state $|0\rangle$.
# 
# **Goals:**
# 
# * Transform the `sum` register into the binary sum (little-engian) of $\phi$ and $\psi$.
# * Transform the `carry` qubit into the carry bit produced by that sum.
# 
# **Challenge:**
# 
# Can you do this without allocating extra qubits?

# In[ ]:


get_ipython().run_line_magic('kata', 'T17_ArbitraryAdder_Test')

operation ArbitraryAdder (a : Qubit[], b : Qubit[], sum : Qubit[], carry : Qubit) : Unit is Adj {
    // ...
}


# ## Part II. Simple In-place Adder
# 
# The adder from the previous section requires empty qubits to accept the result.
# This section adapts the previous adder to calculate the sum in-place,
# that is, to reuse one of the numerical inputs for storing the output.

# ### Task 2.1. In-place summation of two bits
# 
# **Inputs:**
# 
#   1. qubit `a` in an arbitrary state $|\phi\rangle$,
#   2. qubit `b` in an arbitrary state $|\psi\rangle$.
# 
# **Goals:**
# 
# * Transform qubit `b` into the lowest bit of the sum of $\phi$ and $\psi$.
# * Leave qubit `a` unchanged.

# In[ ]:


get_ipython().run_line_magic('kata', 'T21_LowestBitSumInPlace_Test')

operation LowestBitSumInPlace (a : Qubit, b : Qubit) : Unit is Adj {
    // ...
}


# > Can we re-use one of the input bits to calculate the carry in-place as well? Why or why not?

# ### Task 2.2. In-place one-bit adder
# 
# **Inputs:**
# 
#   1. qubit `a` in an arbitrary state $|\phi\rangle$,
#   2. qubit `b` in an arbitrary state $|\psi\rangle$,
#   3. qubit `carry` in state $|0\rangle$.
# 
# **Goals:**
# 
# * Transform the `carry` qubit into the carry bit from the addition of $\phi$ and $\psi$.
# * Transform qubit `b` into the lowest bit of $\phi + \psi$.
# * Leave qubit `a` unchanged.
# 
# <br/>
# <details>
#     <summary>Need a hint? Click here</summary>
#     Think very carefully about the order in which you apply the operations.
# </details>

# In[ ]:


get_ipython().run_line_magic('kata', 'T22_OneBitAdderInPlace_Test')

operation OneBitAdderInPlace (a : Qubit, b : Qubit, carry : Qubit) : Unit is Adj {
    // ...
}


# ### Task 2.3. In-place summation of three bits
# 
# **Inputs:**
# 
#   1. qubit `a` in an arbitrary state $|\phi\rangle$,
#   2. qubit `b` in an arbitrary state $|\psi\rangle$,
#   3. qubit `carryin` in an arbitrary state $|\omega\rangle$.
# 
# **Goals:**
# 
# * Transform qubit `b` into the lowest bit from the sum $\phi + \psi + \omega$.
# * Leave qubits `a` and `carryin` unchanged.

# In[ ]:


get_ipython().run_line_magic('kata', 'T23_HighBitSumInPlace_Test')

operation HighBitSumInPlace (a : Qubit, b : Qubit, carryin : Qubit) : Unit is Adj {
    // ...
}


# ### Task 2.4. In-place two-bit adder
# 
# **Inputs:**
# 
#   1. two-qubit register `a` in an arbitrary state $|\phi\rangle$,
#   2. two-qubit register `b` in an arbitrary state $|\psi\rangle$,
#   3. qubit `carry` in state $|0\rangle$.
# 
# **Goals:**
# 
# * Transform register `b` into the state $|\phi + \psi\rangle$.
# * Transform the `carry` qubit into the carry bit from the addition.
# * Leave register `a` unchanged.

# In[ ]:


get_ipython().run_line_magic('kata', 'T24_TwoBitAdderInPlace_Test')

operation TwoBitAdderInPlace (a : Qubit[], b : Qubit[], carry : Qubit) : Unit is Adj {
    // ...
}


# ### Task 2.5. In-place N-bit adder
# 
# **Inputs:**
# 
#   1. $N$-qubit register `a` in an arbitrary state $|\phi\rangle$,
#   2. $N$-qubit register `b` in an arbitrary state $|\psi\rangle$,
#   3. qubit `carry` in state $|0\rangle$.
# 
# **Goals:**
# 
# * Transform register `b` into the state $|\phi + \psi\rangle$.
# * Transform the `carry` qubit into the carry bit from the addition.
# * Leave register `a` unchanged.

# In[ ]:


get_ipython().run_line_magic('kata', 'T25_ArbitraryAdderInPlace_Test')

operation ArbitraryAdderInPlace (a : Qubit[], b : Qubit[], carry : Qubit) : Unit is Adj {
    // ...
}


# ## Part III*. Improved In-place Adder
# 
# The in-place adder doesn't require quite as many qubits for the inputs and outputs, but it still requires an array of extra ("ancillary") qubits to perform the calculation.
# 
# A relatively recent algorithm allows you to perform the same calculation using only one additional qubit.
# 
# ### Theory
# 
# * [Paper on improved ripple carry addition](https://arxiv.org/pdf/quant-ph/0410184.pdf) by Steven A. Cuccaro, Thomas G. Draper, Samuel A. Kutin, and David Petrie Moulton.

# ### Task 3.1. Majority gate
# 
# **Inputs:**
# 
#   1. qubit `a` in an arbitrary state $|\phi\rangle$,
#   2. qubit `b` in an arbitrary state $|\psi\rangle$,
#   3. qubit `c` in an arbitrary state $|\omega\rangle$.
# 
# **Goal:** Construct the "in-place majority" gate:
# 
# * Transform qubit `a` into the carry bit from the sum $\phi + \psi + \omega$.
# * Transform qubit `b` into $|\phi + \psi\rangle$.
# * Transform qubit `c` into $|\phi + \omega\rangle$.

# In[ ]:


get_ipython().run_line_magic('kata', 'T31_Majority_Test')

operation Majority (a : Qubit, b : Qubit, c : Qubit) : Unit is Adj {
    // ...
}


# ### Task 3.2. UnMajority and Add gate
# 
# **Inputs:**
# 
#   1. qubit `a` storing the carry bit from the sum $\phi + \psi + \omega$,
#   2. qubit `b` in state $|\phi + \psi\rangle$,
#   3. qubit `c` in state $|\phi + \omega\rangle$.
# 
# **Goal:** Construct the "un-majority and add", or "UMA" gate:
# 
# * Restore qubit `a` into state $|\phi\rangle$.
# * Transform qubit `b` into state $|\phi + \psi + \omega\rangle$.
# * Restore qubit `c` into state $|\omega\rangle$.

# In[ ]:


get_ipython().run_line_magic('kata', 'T32_UnMajorityAdd_Test')

operation UnMajorityAdd (a : Qubit, b : Qubit, c : Qubit) : Unit is Adj {
    // ...
}


# ### Task 3.3. One-bit Majority-UMA adder
# 
# **Inputs:**
# 
# 1. qubit `a` in an arbitrary state $|\phi\rangle$,
# 2. qubit `b` in an arbitrary state $|\psi\rangle$,
# 3. qubit `carry` in state $|0\rangle$.
# 
# **Goal:** Construct a one-bit binary adder from task 2.2 using Majority and UMA gates.
# 
# <br/>
# <details>
#     <summary>Need a hint? Click here</summary>
#     Allocate an extra qubit to hold the carry bit used in Majority and UMA gates during the computation. It's less efficient here, but it will be helpful for the next tasks.
# </details>

# In[ ]:


get_ipython().run_line_magic('kata', 'T33_OneBitMajUmaAdder_Test')

operation OneBitMajUmaAdder (a : Qubit, b : Qubit, carry : Qubit) : Unit is Adj {
    // ...
}


# ### Task 3.4. Two-bit Majority-UMA adder
# 
# **Inputs:**
# 
#   1. two-qubit register `a` in an arbitrary state $|\phi\rangle$,
#   2. two-qubit register `b` in an arbitrary state $|\psi\rangle$,
#   3. qubit `carry` in state $|0\rangle$.
# 
# **Goal:** Construct a two-bit binary adder from task 2.4 using Majority and UMA gates.
# 
# <br/>
# <details>
#     <summary>Need a hint? Click here</summary>
#     Think carefully about which qubits you need to pass to the two gates.
# </details>

# In[ ]:


get_ipython().run_line_magic('kata', 'T34_TwoBitMajUmaAdder_Test')

operation TwoBitMajUmaAdder (a : Qubit[], b : Qubit[], carry : Qubit) : Unit is Adj {
    // ...
}


# ### Task 3.5. N-bit Majority-UMA adder
# 
# **Inputs:**
# 
#   1. $N$-qubit register `a` in an arbitrary state $|\phi\rangle$,
#   2. $N$-qubit register `b` in an arbitrary state $|\psi\rangle$,
#   3. qubit `carry` in state $|0\rangle$.
# 
# **Goal:** Construct an N-bit binary adder from task 2.5 using only one ancillary qubit.

# In[ ]:


get_ipython().run_line_magic('kata', 'T35_ArbitraryMajUmaAdder_Test')

operation ArbitraryMajUmaAdder (a : Qubit[], b : Qubit[], carry : Qubit) : Unit is Adj {
    // ...
}


# ## Part IV*. In-place Subtractor
# 
# Subtracting a number is the same operation as adding a negative number.
# As such, it's pretty easy to adapt the adder we just built to act as a subtractor.

# ### Task 4.1. N-bit Subtractor
# 
# **Inputs:**
# 
#   1. $N$-qubit register `a` in an arbitrary state $|\phi\rangle$,
#   2. $N$-qubit register `b` in an arbitrary state $|\psi\rangle$,
#   3. qubit `borrow` in state $|0\rangle$.
# 
# **Goal:** Construct an N-bit binary subtractor.
# 
# * Transform register `b` into the state $|\psi - \phi\rangle$.
# * Set qubit `borrow` to $|1\rangle$ if that subtraction requires a borrow.
# * Leave register `a` unchanged.
# 
# <br/>
# <details>
#     <summary>Need a hint? Click here</summary>
#     Use the adder you already built. Experiment with inverting registers before and after the addition.
# </details>

# In[ ]:


get_ipython().run_line_magic('kata', 'T41_Subtractor_Test')

operation Subtractor (a : Qubit[], b : Qubit[], borrow : Qubit) : Unit is Adj {
    // ...
}

