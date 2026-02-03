# Using Proxies

The `vitallens` client supports proxy configuration to traverse corporate firewalls.

## Standard Network Proxy

Use this if you are behind a corporate firewall that requires a forward proxy to reach the internet. In this mode, the client handles authentication normally, so you must provide your API Key.

```python
from vitallens import VitalLens, Method

proxies = {
  'https': 'http://10.10.1.10:3128',
}

vl = VitalLens(
  method="vitallens", 
  api_key="YOUR_API_KEY", 
  proxies=proxies
)
```
