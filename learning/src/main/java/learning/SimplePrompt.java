package learning;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.time.Duration;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.json.JSONObject;

/**
 * This makes a simple call to a local Ollama instance on its default port. It passes in a prompt, requests data not be
 * streamed, and outputs the results.
 * 
 * I cribbed how to do this from
 * https://dev.to/jayantaadhikary/using-the-ollama-api-to-run-llms-and-generate-responses-locally-18b7 .
 */

public class SimplePrompt {

static Logger logger = LogManager.getLogger(SimplePrompt.class);

public static void main(String[] args) {
    try {

	logger.info("Started");

	JSONObject jsonObject = new JSONObject();
	jsonObject.put("model", "gemma:2b"); // a fast model
	jsonObject.put("stream", false); // will wait for a single reply (default is stream)
	jsonObject.put("prompt", "What is water made of?");

	String payload = jsonObject.toString();

	logger.info("payload: " + payload);

	// an HTTP client that will wait for up to one hour for a response
	HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofHours(1)).build();

	HttpRequest req = HttpRequest.newBuilder(new URI("http://localhost:11434/api/generate"))
		.POST(BodyPublishers.ofString(payload)).build();

	HttpResponse<String> response = client.send(req, BodyHandlers.ofString());

	logger.info("response: " + response);

	String body = response.body();

	JSONObject rc = new JSONObject(body);

	// print out the JSON in a pretty, indented format
	logger.info("body: " + rc.toString(4));

	logger.info("Completed");

    } catch (Exception e) {
	logger.error(e.getMessage(), e);
	System.exit(1);
    }
    System.exit(0);
}

}
