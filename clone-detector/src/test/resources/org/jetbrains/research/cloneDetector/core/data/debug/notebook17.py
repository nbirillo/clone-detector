#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pylabnet.hardware.filterwheel.filterwheel import FW102CFilterWheel
from pylabnet.utils.logging.logger import LogClient
from pylabnet.core.generic_server import GenericServer


# #  Instantiate Logger

# In[2]:


filterwheel_logger = LogClient(
    host='localhost', 
    port=1399, 
    module_tag='Filterwheel Log Server'
)


# # Instantiate Filterwheel Connection

# In[3]:


device_name = 'Filterwheel 1'
port_name = 'COM10'
filters = {
    '1' : 'BP 740',
    '2' : '0 ND',
    '3' : '0.4 ND',
    '4' : '1 ND',
    '5' : '3 ND',
    '6' : '4 ND',
}

filterwheel = FW102CFilterWheel(port_name=port_name, device_name=device_name, filters=filters, logger=filterwheel_logger)


# # Instantiate Client Server

# In[4]:


filterwheel_service = filterwheel.Service()
filterwheel_service.assign_module(module=filterwheel)
filterwheel_service.assign_logger(logger=filterwheel_logger)
filterwheel_service_server = GenericServer(
    service=filterwheel_service, 
    host='localhost', 
    port=5698
)


# In[5]:


filterwheel_service_server.start()


# # Watch Connections

# In[ ]:


filterwheel.close()

