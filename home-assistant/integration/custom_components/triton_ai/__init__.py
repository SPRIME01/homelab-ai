"""
TritonAI integration for Home Assistant.
Connects Home Assistant with Triton Inference Server and Ray for AI capabilities.
"""
import asyncio
import logging
import voluptuous as vol
from datetime import timedelta

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.event import async_track_time_interval

from .const import (
    DOMAIN,
    CONF_TRITON_URL,
    CONF_RAY_ADDRESS,
    CONF_MODELS,
    CONF_SENSOR_ANALYSIS_INTERVAL,
    CONF_LOG_LEVEL,
    DEFAULT_SENSOR_ANALYSIS_INTERVAL,
    DEFAULT_LOG_LEVEL,
    PLATFORMS,
)
from .triton_client import TritonClient
from .ray_manager import RayTaskManager
from .sensor_analysis import SensorAnalysisService

_LOGGER = logging.getLogger(__name__)

# Configuration schema
MODEL_SCHEMA = vol.Schema({
    vol.Required(str): cv.string,
})

CONFIG_SCHEMA = vol.Schema({
    DOMAIN: vol.Schema({
        vol.Required(CONF_TRITON_URL): cv.url,
        vol.Required(CONF_RAY_ADDRESS): cv.string,
        vol.Required(CONF_MODELS): MODEL_SCHEMA,
        vol.Optional(CONF_SENSOR_ANALYSIS_INTERVAL, default=DEFAULT_SENSOR_ANALYSIS_INTERVAL):
            cv.positive_int,
        vol.Optional(CONF_LOG_LEVEL, default=DEFAULT_LOG_LEVEL):
            vol.In(["debug", "info", "warning", "error", "critical"]),
    })
}, extra=vol.ALLOW_EXTRA)

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the TritonAI integration."""
    if DOMAIN not in config:
        return True

    domain_config = config[DOMAIN]

    # Configure logging
    log_level = domain_config[CONF_LOG_LEVEL]
    logging.getLogger(__name__).setLevel(getattr(logging, log_level.upper()))

    # Initialize the data structure
    hass.data[DOMAIN] = {
        "config": domain_config,
        "clients": {},
        "services": {},
    }

    # Initialize Triton client
    triton_url = domain_config[CONF_TRITON_URL]
    triton_client = TritonClient(triton_url)

    success = await triton_client.initialize()
    if not success:
        _LOGGER.error("Failed to initialize Triton client")
        return False

    hass.data[DOMAIN]["clients"]["triton"] = triton_client

    # Initialize Ray task manager
    ray_address = domain_config[CONF_RAY_ADDRESS]
    ray_manager = RayTaskManager(ray_address)

    success = await ray_manager.initialize()
    if not success:
        _LOGGER.error("Failed to initialize Ray task manager")
        return False

    hass.data[DOMAIN]["clients"]["ray"] = ray_manager

    # Set up sensor analysis service
    analysis_interval = domain_config[CONF_SENSOR_ANALYSIS_INTERVAL]
    sensor_analysis = SensorAnalysisService(hass, triton_client, ray_manager)

    await sensor_analysis.initialize()
    hass.data[DOMAIN]["services"]["sensor_analysis"] = sensor_analysis

    # Schedule regular sensor analysis
    async def analyze_sensors_regularly(_now=None):
        """Run sensor analysis on a schedule."""
        await sensor_analysis.analyze_all()

    async_track_time_interval(
        hass, analyze_sensors_regularly, timedelta(minutes=analysis_interval)
    )

    # Register platforms
    for platform in PLATFORMS:
        hass.async_create_task(
            hass.helpers.discovery.async_load_platform(
                platform, DOMAIN, {}, config
            )
        )

    # Register services
    from .services import register_services
    register_services(hass)

    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up TritonAI from a config entry."""
    # Extract config
    config = {
        CONF_TRITON_URL: entry.data[CONF_TRITON_URL],
        CONF_RAY_ADDRESS: entry.data[CONF_RAY_ADDRESS],
        CONF_MODELS: entry.data[CONF_MODELS],
        CONF_SENSOR_ANALYSIS_INTERVAL: entry.data.get(
            CONF_SENSOR_ANALYSIS_INTERVAL, DEFAULT_SENSOR_ANALYSIS_INTERVAL
        ),
        CONF_LOG_LEVEL: entry.data.get(
            CONF_LOG_LEVEL, DEFAULT_LOG_LEVEL
        ),
    }

    # Store config in hass.data
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}
    hass.data[DOMAIN]["config"] = config

    # Forward to regular setup
    for platform in PLATFORMS:
        hass.async_create_task(
            hass.config_entries.async_forward_entry_setup(entry, platform)
        )

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Unload platforms
    unload_ok = all(
        await asyncio.gather(
            *[
                hass.config_entries.async_forward_entry_unload(entry, platform)
                for platform in PLATFORMS
            ]
        )
    )

    # Clean up resources
    if unload_ok:
        triton_client = hass.data[DOMAIN]["clients"].get("triton")
        if triton_client:
            await triton_client.close()

        ray_manager = hass.data[DOMAIN]["clients"].get("ray")
        if ray_manager:
            await ray_manager.close()

        sensor_analysis = hass.data[DOMAIN]["services"].get("sensor_analysis")
        if sensor_analysis:
            await sensor_analysis.close()

        hass.data[DOMAIN].pop(entry.entry_id)

    return unload_ok
