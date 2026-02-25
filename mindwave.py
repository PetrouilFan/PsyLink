'''
Mindwave Mobile Serial Driver for Python 3.x

This driver implements the serial protocol that is being used in the Mindwave Mobile Headset.

 OfflineHeadset: this class can be used in the same as Headset to replay a previous stored file.

'''
import select, serial, threading
from pprint import pprint
import time
import datetime
import os
import logging

logger = logging.getLogger(__name__)

from enum import Enum

class ByteCode(Enum):
    CONNECT              = b'\xc0'
    DISCONNECT           = b'\xc1'
    AUTOCONNECT          = b'\xc2'
    SYNC                 = b'\xaa'
    EXCODE               = 0x55
    POOR_SIGNAL          = 0x02
    ATTENTION            = 0x04
    MEDITATION           = 0x05
    BLINK                = 0x16
    HEADSET_CONNECTED    = b'\xd0'
    HEADSET_NOT_FOUND    = b'\xd1'
    HEADSET_DISCONNECTED = b'\xd2'
    REQUEST_DENIED       = b'\xd3'
    STANDBY_SCAN         = b'\xd4'
    RAW_VALUE            = 0x80
    ASIC_EEG_POWER       = b'\x83'

class Status(Enum):
    CONNECTED     = 'connected'
    SCANNING      = 'scanning'
    STANDBY       = 'standby'

# Use me to playback previous recorded files as if they were recorded now.
# (using the same python class)
class OfflineHeadset:
    """
    An Offline MindWave Headset
    """
    def __init__(self, filename):
        self.basefilename = filename
        self.readcounter = 0
        self.running = True
        self.fileindex = 0
        self.f = None
        self.poor_signal = 1

    def setup(self):
        pass

    def setupfile(self):
        self.datasetfile = self.basefilename
        logger.info(f"Opening dataset file: {self.datasetfile}")
        if os.path.isfile(self.datasetfile):
            if self.f:
                self.f.close()
            self.f = open(self.datasetfile,'r')
            return True
        else:
            return False

    def nextline(self):
        line = None
        if self.f:
            line = self.f.readline()
        if (not line):
            self.fileindex = self.fileindex + 1

            if self.setupfile():
                return self.nextline()
            else:
                return None
        else:
            return line

    def dequeue(self):
        line = self.nextline()
        if (line):
            data = line.split('\r\n')[0].split(' ')
            self.raw_value = data[1]
            self.attention = data[2]
            self.meditation = data[3]
            self.blink = data[4]

            self.readcounter = self.readcounter + 1
            return self
        else:
            self.running = False
            return None


    def close(self):
        if (self.f):
            self.f.close()

    def stop(self):
        self.close()


class Headset(object):
    """
    A MindWave Headset
    """

    class DongleListener(threading.Thread):
        """
        Serial listener for dongle device.
        """
        def __init__(self, headset, *args, **kwargs):
            """Set up the listener device."""
            self.headset = headset
            self.counter = 0
            super(Headset.DongleListener, self).__init__(*args, **kwargs)

        def run(self):
            """Run the listener thread."""
            s = self.headset.dongle

            self.headset.running = True

            # Re-apply settings to ensure packet stream
            s.write(ByteCode.DISCONNECT.value)
            d = s.getSettingsDict()
            for i in range(2):
                d['rtscts'] = not d['rtscts']
                s.applySettingsDict(d)

            while self.headset.running:
                # Begin listening for packets
                try:
                    if s.read() == ByteCode.SYNC.value and s.read() == ByteCode.SYNC.value:
                        # Packet found, determine plength
                        while True:
                            plength = ord(s.read())
                            if plength != 170:
                                break
                        if plength > 170:
                            continue

                        # Read in the payload
                        payload = s.read(plength)

                        # Verify its checksum
                        val = sum(b for b in payload[:-1])
                        val &= 0xff
                        val = ~val & 0xff
                        chksum = ord(s.read())

                        if val == chksum:
                            self.parse_payload(payload)
                        else:
                            logger.warning("Bad checksum: expected 0x%02x, got 0x%02x", chksum, val)
                except (select.error, OSError):
                    break
                except serial.SerialTimeoutException:
                    logger.warning("Serial timeout, waiting...")
                    continue
                except serial.SerialException as e:
                    logger.error(f"Serial exception: {e}, attempting reconnect in 2 seconds...")
                    s.close()
                    time.sleep(2)
                    try:
                        self.headset.serial_open()
                        s = self.headset.dongle
                        s.write(ByteCode.DISCONNECT.value)
                        d = s.getSettingsDict()
                        for i in range(2):
                            d['rtscts'] = not d['rtscts']
                            s.applySettingsDict(d)
                    except Exception as re_err:
                        logger.error(f"Reconnect failed: {re_err}")
                        break


            logger.info('Closing connection...')
            if s and s.isOpen():
                s.close()

        def parse_payload(self, payload):
            """Parse the payload to extract data and fire events."""
            while len(payload) > 0:
                code, payload = payload[0], payload[1:]

                if code == ByteCode.EXCODE.value:
                    # Count the number of EXCODEs
                    excode_count = 1
                    while len(payload) > 0 and payload[0] == ByteCode.EXCODE.value:
                        excode_count += 1
                        payload = payload[1:]
                    continue
                
                if code < 0x80:
                    # This is a single-byte code
                    if len(payload) == 0:
                        logger.warning("Payload parsing dropped: expecting single byte value but buffer is empty")
                        break
                    value, payload = payload[0], payload[1:]

                    if code == ByteCode.POOR_SIGNAL.value:
                        old_poor_signal = self.headset.poor_signal
                        self.headset.poor_signal = value
                        if self.headset.poor_signal > 0:
                            if old_poor_signal == 0:
                                for handler in self.headset.poor_signal_handlers:
                                    handler(self.headset, self.headset.poor_signal)
                        else:
                            if old_poor_signal > 0:
                                for handler in self.headset.good_signal_handlers:
                                    handler(self.headset, self.headset.poor_signal)
                    elif code == ByteCode.ATTENTION.value:
                        self.headset.attention = value
                        for handler in self.headset.attention_handlers:
                            handler(self.headset, self.headset.attention)
                    elif code == ByteCode.MEDITATION.value:
                        self.headset.meditation = value
                        for handler in self.headset.meditation_handlers:
                            handler(self.headset, self.headset.meditation)
                    elif code == ByteCode.BLINK.value:
                        self.headset.blink = value
                        for handler in self.headset.blink_handlers:
                            handler(self.headset, self.headset.blink)
                    else:
                        logger.debug(f"Unhandled single-byte code: {hex(code)}")
                else:
                    # This is a multi-byte code
                    if len(payload) == 0:
                        logger.warning("Payload parsing dropped: expecting length byte but buffer is empty")
                        break
                    
                    vlength, payload = payload[0], payload[1:]
                    
                    if len(payload) < vlength:
                        logger.warning("Payload parsing dropped: vlength specifies larger chunk than buffer holds")
                        break
                        
                    value, payload = payload[:vlength], payload[vlength:]

                    if code == ByteCode.RAW_VALUE.value and len(value) >= 2:
                        raw=value[0]*256+value[1]
                        if (raw>=32768):
                            raw=raw-65536
                        self.headset.raw_value = raw
                        for handler in self.headset.raw_value_handlers:
                            handler(self.headset, self.headset.raw_value)
                    elif code == ByteCode.HEADSET_CONNECTED.value:
                        run_handlers = self.headset.status != Status.CONNECTED.value
                        self.headset.status = Status.CONNECTED.value
                        self.headset.headset_id = value.hex() if isinstance(value, bytes) else value.encode('hex')
                        if run_handlers:
                            for handler in \
                                self.headset.headset_connected_handlers:
                                handler(self.headset)
                    elif code == ByteCode.HEADSET_NOT_FOUND.value:
                        if vlength > 0:
                            not_found_id = value.hex() if isinstance(value, bytes) else value.encode('hex')
                            for handler in \
                                self.headset.headset_notfound_handlers:
                                handler(self.headset, not_found_id)
                        else:
                            for handler in \
                                self.headset.headset_notfound_handlers:
                                handler(self.headset, None)
                    elif code == ByteCode.HEADSET_DISCONNECTED.value:
                        headset_id = value.hex() if isinstance(value, bytes) else value.encode('hex')
                        for handler in \
                            self.headset.headset_disconnected_handlers:
                            handler(self.headset, headset_id)
                    elif code == ByteCode.REQUEST_DENIED.value:
                        # Request denied
                        for handler in self.headset.request_denied_handlers:
                            handler(self.headset)
                    elif code == ByteCode.STANDBY_SCAN.value:
                        # Standby/Scan mode
                        self.headset.status = Status.SCANNING.value
                        try:
                            byte = value[0] if isinstance(value, bytes) else ord(value[0])
                        except IndexError:
                            byte = None
                        if byte:
                            run_handlers = (self.headset.status !=
                                            Status.SCANNING.value)
                            self.headset.status = Status.SCANNING.value
                            if run_handlers:
                                for handler in self.headset.scanning_handlers:
                                    handler(self.headset)
                        else:
                            run_handlers = (self.headset.status !=
                                            Status.STANDBY.value)
                            self.headset.status = Status.STANDBY.value
                            if run_handlers:
                                for handler in self.headset.standby_handlers:
                                    handler(self.headset)
                    elif code == ByteCode.ASIC_EEG_POWER.value:
                        j = 0
                        for i in ['delta', 'theta', 'low-alpha', 'high-alpha', 'low-beta', 'high-beta', 'low-gamma', 'mid-gamma']:
                            v0 = value[j] if isinstance(value, bytes) else ord(value[j])
                            v1 = value[j+1] if isinstance(value, bytes) else ord(value[j+1])
                            v2 = value[j+2] if isinstance(value, bytes) else ord(value[j+2])
                            
                            # Fixed math base: 256 for bytes instead of 255
                            self.headset.waves[i] = (v0 << 16) | (v1 << 8) | v2
                            j += 3
                        for handler in self.headset.waves_handlers:
                            handler(self.headset, self.headset.waves)

    def __init__(self, device, headset_id=None, open_serial=True):
        """Initialize the headset."""
        # Initialize headset values
        self.dongle = None
        self.listener = None
        self.device = device
        self.headset_id = headset_id
        self.poor_signal = 255
        self.attention = 0
        self.meditation = 0
        self.blink = 0
        self.raw_value = 0
        self.waves = {}
        self.status = None
        self.count = 0
        self.running = False

        # Create event handler lists
        self.poor_signal_handlers = []
        self.good_signal_handlers = []
        self.attention_handlers = []
        self.meditation_handlers = []
        self.blink_handlers = []
        self.raw_value_handlers = []
        self.waves_handlers = []
        self.headset_connected_handlers = []
        self.headset_notfound_handlers = []
        self.headset_disconnected_handlers = []
        self.request_denied_handlers = []
        self.scanning_handlers = []
        self.standby_handlers = []

        # Open the socket
        if open_serial:
            self.serial_open()

    def connect(self, headset_id=None):
        """Connect to the specified headset id."""
        if headset_id:
            self.headset_id = headset_id
        else:
            headset_id = self.headset_id
            if not headset_id:
                self.autoconnect()
                return
        
        id_bytes = bytes.fromhex(headset_id) if hasattr(bytes, 'fromhex') else headset_id.decode('hex')
        self.dongle.write(ByteCode.CONNECT.value + id_bytes)

    def autoconnect(self):
        """Automatically connect device to headset."""
        self.dongle.write(ByteCode.AUTOCONNECT.value)

    def disconnect(self):
        """Disconnect the device from the headset."""
        self.dongle.write(DISCONNECT)

    def serial_open(self):
        """Open the serial connection and begin listening for data."""
        # Establish serial connection to the dongle
        if not self.dongle or not self.dongle.isOpen():
            self.dongle = serial.Serial(self.device, 115200, timeout=1.0)

        # Begin listening to the serial device
        if not self.listener or not self.listener.isAlive():
            self.listener = self.DongleListener(self)
            self.listener.daemon = True
            self.listener.start()

    def serial_close(self):
        """Close the serial connection."""
        self.dongle.close()

    def stop(self):
        self.running = False
