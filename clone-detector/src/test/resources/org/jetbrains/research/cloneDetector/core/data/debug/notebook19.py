#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[1]:


# general setup
get_ipython().run_line_magic('run', 'setup.ipynb')
# import popgen
# development setup
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
get_ipython().run_line_magic('aimport', 'popgen')
get_ipython().run_line_magic('aimport', 'popgen.config')
get_ipython().run_line_magic('aimport', 'popgen.analysis')
get_ipython().run_line_magic('aimport', 'popgen.util')
get_ipython().run_line_magic('aimport', 'popgen.caching')


# In[2]:


# setup analysis (see analysis_config.ipynb for configuration)
analysis = popgen.analysis.PopulationAnalysis('../data/analysis')
analysis


# In[3]:


# plotting setup
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('paper')
plt.rcParams['figure.dpi'] = 120
get_ipython().run_line_magic('config', "InlineBackend.figure_formats = {'retina', 'png'}")


# ## Nucleotide diversity

# In[4]:


chromosomes = ('2R', '2L', '3R', '3L', 'X')
autosomes = chromosomes[:4]


# In[5]:


analysis.windowed_statistic_distplot('diversity', chrom=autosomes, window_size=100000, pop='junju', xlim=(0, 1.6))


# In[6]:


analysis.windowed_statistic_distplot('diversity', chrom=autosomes, window_size=100000, pop='mbogolo', xlim=(0, 1.6))


# In[7]:


fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
analysis.windowed_statistic_violinplot('diversity', chrom=autosomes, window_size=100000, pops=['junju', 'mbogolo'], ax=axs[0], ylim=(0, 1.6))
analysis.windowed_statistic_violinplot('diversity', chrom='X', window_size=100000, pops=['junju', 'mbogolo'], ax=axs[1], ylim=(0, 1.6))
axs[1].set_xlabel('')
axs[1].set_ylabel('')
fig.tight_layout()
fig.savefig('../artwork/pi_comparisons.png', bbox_inches='tight')


# In[8]:


analysis.windowed_statistic_compare('diversity', chrom=autosomes, pops=['junju', 'mbogolo'], window_size=100000)


# In[9]:


analysis.windowed_statistic_compare('diversity', chrom='X', pops=['junju', 'mbogolo'], window_size=100000)


# In[10]:


fig = plt.figure(figsize=(8, 2))
analysis.windowed_statistic_genomeplot('diversity', chroms=chromosomes, pop='junju', window_size=100000, fig=fig)
analysis.windowed_statistic_genomeplot('diversity', chroms=chromosomes, pop='mbogolo', window_size=100000, fig=fig)
fig.savefig('../artwork/pi_genome.png', bbox_inches='tight')


# In[11]:


fig = plt.figure(figsize=(8, 2))
analysis.windowed_diversity_delta_genomeplot(chroms=chromosomes, pop1='junju', pop2='mbogolo', window_size=100000, fig=fig)
fig.savefig('../artwork/pi_delta_genome.png', bbox_inches='tight')


# ## Watterson's theta

# In[12]:


analysis.windowed_statistic_distplot('watterson_theta', chrom=autosomes, window_size=100000, pop='junju', xlim=(0, 1.6))


# In[13]:


analysis.windowed_statistic_distplot('watterson_theta', chrom=autosomes, window_size=100000, pop='mbogolo', xlim=(0, 1.6))


# In[14]:


fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
analysis.windowed_statistic_violinplot('watterson_theta', chrom=autosomes, window_size=100000, pops=['junju', 'mbogolo'], ax=axs[0], ylim=(0, 1.6))
analysis.windowed_statistic_violinplot('watterson_theta', chrom='X', window_size=100000, pops=['junju', 'mbogolo'], ax=axs[1], ylim=(0, 1.6))
axs[1].set_xlabel('')
axs[1].set_ylabel('')
fig.tight_layout()
fig.savefig('../artwork/theta_w_comparisons.png', bbox_inches='tight')


# In[15]:


analysis.windowed_statistic_compare('watterson_theta', chrom=autosomes, pops=['junju', 'mbogolo'], window_size=100000)


# In[16]:


analysis.windowed_statistic_compare('watterson_theta', chrom='X', pops=['junju', 'mbogolo'], window_size=100000)


# In[17]:


fig = plt.figure(figsize=(8, 2))
analysis.windowed_statistic_genomeplot('watterson_theta', chroms=chromosomes, pop='junju', window_size=100000, fig=fig)
analysis.windowed_statistic_genomeplot('watterson_theta', chroms=chromosomes, pop='mbogolo', window_size=100000, fig=fig)
fig.savefig('../artwork/theta_w_genome.png', bbox_inches='tight')


# ## Tajima's D

# In[18]:


fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
analysis.windowed_statistic_violinplot('tajima_d', chrom=autosomes, window_size=100000, pops=['junju', 'mbogolo'], ax=axs[0])
analysis.windowed_statistic_violinplot('tajima_d', chrom='X', window_size=100000, pops=['junju', 'mbogolo'], ax=axs[1])
axs[1].set_xlabel('')
axs[1].set_ylabel('')
fig.tight_layout()
fig.savefig('../artwork/tajima_d_comparisons.png', bbox_inches='tight')


# In[19]:


analysis.windowed_statistic_compare('tajima_d', chrom=autosomes, pops=['junju', 'mbogolo'], window_size=100000)


# In[20]:


analysis.windowed_statistic_compare('tajima_d', chrom='X', pops=['junju', 'mbogolo'], window_size=100000)


# In[21]:


fig = plt.figure(figsize=(8, 2))
analysis.windowed_statistic_genomeplot('tajima_d', chroms=chromosomes, pop='junju', window_size=100000, fig=fig)
analysis.windowed_statistic_genomeplot('tajima_d', chroms=chromosomes, pop='mbogolo', window_size=100000, fig=fig)
fig.savefig('../artwork/tajima_d_genome.png', bbox_inches='tight')

