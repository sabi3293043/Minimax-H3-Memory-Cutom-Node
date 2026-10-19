"""Optional native observer status events; importing never needs a web server."""
import logging

LOG = logging.getLogger(__name__)


def notify_native(node_id, message, *, log=False, **stats):
    if log:
        LOG.warning('[MosaiChunk H3 native] %s', message)
    if node_id is None:
        return
    try:
        from server import PromptServer
        if PromptServer.instance is not None:
            PromptServer.instance.send_sync('mosaichunk.native_status',
                dict(node=node_id, message=message, **stats))
    except Exception:
        LOG.debug('Native observer status delivery failed', exc_info=True)
