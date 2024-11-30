"""(Very) Simple Implementation of Artnet.

Python Version: 3.6
Source: http://artisticlicence.com/WebSiteMaster/User%20Guides/art-net.pdf

NOTES
- For simplicity: NET and SUBNET not used by default but optional

"""

import socket
import _thread
from time import sleep
from ArtnetUtils import shift_this, put_in_range


class StupidArtnet():
    """(Very) simple implementation of Artnet."""

    def __init__(self, target_ip='127.0.0.1', universe=0, packet_size=512, fps=30,
                 even_packet_size=True, broadcast=False, source_address=None, artsync=False):
        print("init")


    def __del__(self):
        print("__del__")


    def __str__(self):
        return "__str__"


    def make_artdmx_header(self):
        print("make_artdmx_header")


    def make_artsync_header(self):
         print("make_artsync_header")


    def send_artsync(self):
        print("send_artsync")


    def show(self):
        None


    def close(self):
        print("close")
    # THREADING #

    def start(self):
        print("start")


    def stop(self):
        print("stop")

    # SETTERS - HEADER #

    def set_universe(self, universe):
        print("universe:" , universe)


    def set_subnet(self, sub):
        print("sub:" , sub)


    def set_net(self, net):
        print("net:" , net)


    def set_packet_size(self, packet_size):
        print("packet_size:" , packet_size)

    # SETTERS - DATA #

    def clear(self):
        print("clear")


    def set(self, value):
        print("value:")


    def set_16bit(self, address, value, high_first=False):
        print("set_16bit:" , address , value , high_first)


    def set_single_value(self, address, value):
        print("address:" , address , "value:" , value)


    # AUX Function #

    def send(self, packet):
        print("send")


    def set_simplified(self, simplify):
        print("set_simplified")


    def see_header(self):
        """Show header values."""
        print("see_header")


    def see_buffer(self):
        """Show buffer values."""
        print("see_buffer")


    def blackout(self):
        print("blackout")


    def flash_all(self, delay=None):
        print("flash_all")

