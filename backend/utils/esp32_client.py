import asyncio
import websockets
import json
import logging
from typing import Dict, Any, Optional, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("esp32_client")

class ESP32Client:
    def __init__(self, esp_uri: str = "ws://192.168.137.229:80"):
        self.esp_uri = esp_uri
        self.websocket = None
        self.connected = False
        self.reconnect_task = None
        self.message_handlers = []
        self.shoot_ack_event = asyncio.Event()
        self.last_shoot_ack_message: Optional[Dict[str, Any]] = None
        
    async def connect(self):
        """Connect to the ESP32 WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.esp_uri)
            self.connected = True
            logger.info(f"Connected to ESP32 at {self.esp_uri}")
            
            # Start the message receiver
            asyncio.create_task(self._receive_messages())
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ESP32: {e}")
            self.connected = False
            return False
            
    async def ensure_connection(self):
        """Make sure we're connected to the ESP32"""
        if not self.connected:
            return await self.connect()
        return True
    
    async def send_command(self, command: str):
        """Send a command to the ESP32"""
        if not await self.ensure_connection():
            logger.error("Cannot send command - not connected to ESP32")
            return False
            
        try:
            await self.websocket.send(command)
            logger.info(f"Sent command to ESP32: {command}")
            return True
        except Exception as e:
            logger.error(f"Failed to send command to ESP32: {e}")
            self.connected = False
            await self._schedule_reconnect()
            return False
    
    async def send_json(self, data: Dict[str, Any]):
        """Send JSON data to the ESP32"""
        return await self.send_command(json.dumps(data))

    async def send_json_for_shoot_and_wait_ack(self, data: Dict[str, Any], timeout: float = 5.0) -> bool:
        """Sends JSON data (expected to be a SHOOT command) and waits for a specific SHOOT ACK."""
        if not await self.ensure_connection():
            logger.error("Cannot send SHOOT command - not connected to ESP32")
            return False
            
        command_to_send = json.dumps(data)
        self.shoot_ack_event.clear() 
        self.last_shoot_ack_message = None

        try:
            await self.websocket.send(command_to_send)
            logger.info(f"Sent SHOOT command to ESP32: {command_to_send}, awaiting ACK...")
            
            await asyncio.wait_for(self.shoot_ack_event.wait(), timeout=timeout)
            logger.info(f"SHOOT ACK event received. ACK message: {self.last_shoot_ack_message}")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for SHOOT ACK from ESP32 after {timeout}s.")
            return False
        except websockets.exceptions.ConnectionClosed:
            logger.error("Connection closed while sending SHOOT command or waiting for ACK.")
            self.connected = False
            await self._schedule_reconnect()
            return False
        except Exception as e:
            logger.error(f"Failed to send SHOOT command or wait for ACK: {e}")
            return False
    
    async def _receive_messages(self):
        """Background task to receive messages from ESP32"""
        if not self.websocket:
            return
            
        try:
            async for message_str in self.websocket:
                logger.info(f"Received from ESP32: {message_str}")
                try:
                    message_data = json.loads(message_str)
                    if isinstance(message_data, dict) and \
                       message_data.get("type") == "ack" and \
                       message_data.get("command") == "SHOOT":
                        logger.info("Received SHOOT ACK from ESP32.")
                        self.last_shoot_ack_message = message_data
                        self.shoot_ack_event.set()
                        continue # ACK handled, don't pass to generic handlers
                except json.JSONDecodeError:
                    logger.debug("Received non-JSON message from ESP32 or message not an ACK.")
                
                for handler in self.message_handlers:
                    await handler(message_str) 
        except websockets.exceptions.ConnectionClosed:
            logger.warning("ESP32 WebSocket connection closed")
            self.connected = False
            await self._schedule_reconnect()
        except Exception as e:
            logger.error(f"Error in ESP32 message receiver: {e}")
            self.connected = False
            await self._schedule_reconnect()
    
    async def _schedule_reconnect(self):
        """Schedule a reconnection attempt"""
        if self.reconnect_task is None or self.reconnect_task.done():
            self.reconnect_task = asyncio.create_task(self._reconnect())
    
    async def _reconnect(self, delay: int = 5):
        """Try to reconnect after a delay"""
        await asyncio.sleep(delay)
        await self.connect()
    
    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.connected = False
        
    def add_message_handler(self, handler: Callable):
        """Add a handler function for incoming messages"""
        self.message_handlers.append(handler)
        
    def remove_message_handler(self, handler: Callable):
        """Remove a message handler"""
        if handler in self.message_handlers:
            self.message_handlers.remove(handler)

# Global ESP32 client instance
esp32_client = ESP32Client()
