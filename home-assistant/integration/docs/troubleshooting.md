# Troubleshooting the Triton AI Integration

This guide provides solutions for common issues with the Triton AI integration for Home Assistant.

## Connection Issues

### Cannot Connect to Triton Inference Server

**Symptoms:**
- Error message "Failed to initialize Triton client"
- Integration fails to start or remains offline

**Possible Solutions:**
1. **Check server availability:**
   ```bash
   curl -v http://your-triton-url:8000/v2/health/ready
   ```
   This should return HTTP 200 OK if the server is running.

2. **Check network connectivity:**
   - Ensure Home Assistant can reach the Triton server
   - Check that any necessary network policies or firewall rules allow traffic
   - Verify the URL format (http/https, port number)

3. **Check Triton server logs:**
   ```bash
   kubectl logs -n ai deployment/triton-inference-server
   ```

### Cannot Connect to Ray Cluster

**Symptoms:**
- Error message "Failed to initialize Ray task manager"
- Ray-related features don't work

**Possible Solutions:**
1. **Check Ray cluster status:**
   ```bash
   kubectl get rayclusters -n ai
   kubectl describe rayservice ray-cluster -n ai
   ```

2. **Check Ray head node logs:**
   ```bash
   kubectl logs -n ai deployment/ray-head
   ```

3. **Verify Ray address format:**
   - Should be `ray://host:port` format
   - Typically `ray://ray-head.ai.svc.cluster.local:10001`

## Model Issues

### Models Not Found

**Symptoms:**
- Error message "Model not found" when using services
- Inference requests fail

**Possible Solutions:**
1. **Check available models:**
   ```bash
   curl http://your-triton-url:8000/v2/models
   ```

2. **Verify model repository structure:**
   ```bash
   kubectl exec -it -n ai deployment/triton-inference-server -- ls -la /models
   ```

3. **Update configuration:**
   - Make sure model names in configuration exactly match those on the server
   - Check model version numbers if specified

### Slow Inference or Timeouts

**Symptoms:**
- Services take a long time to complete
- Timeout errors in log
- "Inference request failed" errors

**Possible Solutions:**
1. **Check resource usage:**
   ```bash
   kubectl top pods -n ai
   ```

2. **Adjust model configuration:**
   - Reduce batch size
   - Enable dynamic batching
   - Use model instances for concurrency

3. **Optimize models:**
   - Consider quantizing models (FP16, INT8)
   - Use optimized backends (TensorRT)

4. **Increase timeouts:**
   - In the configuration options, increase service timeout values

## Sensor Analysis Issues

### No Anomalies Detected

**Symptoms:**
- Anomaly sensors always show "off"
- No anomaly events are triggered

**Possible Solutions:**
1. **Check sensor data collection:**
   - Verify that sensor data is being collected
   - Check history length (need enough history points)

2. **Adjust sensitivity:**
   - Lower anomaly thresholds in model configuration

3. **Check model compatibility:**
   - Ensure anomaly detection model works with your sensor types

### Predictions Not Updating

**Symptoms:**
- Prediction sensors show no values or don't update
- No prediction events are fired

**Possible Solutions:**
1. **Check data collection:**
   - Verify sensor data is being collected
   - Check for proper numeric conversion

2. **Check model input requirements:**
   - Some forecasting models need minimum history length
   - Verify data format matches model expectations

## Voice Command Issues

### Speech Recognition Failures

**Symptoms:**
- Voice commands not recognized
- Empty transcription results

**Possible Solutions:**
1. **Check audio format:**
   - Make sure audio is in supported format (WAV, 16kHz, mono)
   - Check audio quality and volume levels

2. **Check model compatibility:**
   - Verify Whisper model is properly configured
   - Check language settings

3. **Troubleshoot audio path:**
   - Verify file path is accessible by Home Assistant
   - Check file permissions

### Intent Recognition Failures

**Symptoms:**
- Speech is transcribed but intent not recognized
- Intent always comes back as "unknown"

**Possible Solutions:**
1. **Check LLM prompt:**
   - Review system prompt for intent recognition
   - Ensure format matches expected output

2. **Check LLM configuration:**
   - Try lower temperature for more deterministic responses
   - Ensure response format is set to JSON if available

3. **Log raw responses:**
   - Enable debug logging to see raw model outputs
   - Check for JSON parsing issues

## Performance Optimization

### Reducing Resource Usage

If your Jetson AGX Orin is overloaded:

1. **Increase analysis intervals:**
   - Set longer intervals between sensor analyses
   - Use triggers rather than polling where possible

2. **Optimize model loading:**
   - Use model control to load/unload models as needed
   - Share model instances between similar functions

3. **Use quantized models:**
   - Use 4-bit or 8-bit quantized models instead of FP16/FP32
   - Enable TensorRT optimization where possible

### Improving Response Times

For faster responses:

1. **Use concurrency:**
   - Configure model instances for concurrent execution
   - Use separate profiles for different latency requirements

2. **Optimize Ray tasks:**
   - Configure Ray to prioritize interactive tasks
   - Use resource groups to isolate workloads

3. **Cache common results:**
   - Enable response caching for common queries
   - Set appropriate TTL values based on use case

## Common Error Messages

### "Cannot connect to Triton server"
- Check URL, network connectivity, and server status

### "Model unavailable"
- Check model repository and model configuration

### "Out of memory"
- Reduce batch size, use quantized models, or free GPU memory

### "Timeout waiting for result"
- Increase timeout values or optimize model performance

### "Failed to parse JSON response"
- Check model output format and system prompt configuration

## Getting Help

If you're still having issues:

1. **Enable debug logging:**
   - Set log_level to "debug" in configuration
   - Check Home Assistant logs for detailed information

2. **Collect diagnostics:**
   ```bash
   kubectl logs -n ai deployment/triton-inference-server > triton.log
   kubectl logs -n ai deployment/ray-head > ray.log
   ```

3. **Open an issue:**
   - Include logs, configuration, and steps to reproduce
   - Describe your hardware and software versions
