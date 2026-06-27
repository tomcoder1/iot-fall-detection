// mDNS keeps working when the Raspberry Pi receives a different DHCP address.
const String piAddress = 'raspberrypi.local:8000';
const String piHost = 'http://$piAddress';
const String wsHost = 'ws://$piAddress/ws';
