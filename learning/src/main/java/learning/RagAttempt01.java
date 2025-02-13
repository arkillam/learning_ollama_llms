package learning;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class RagAttempt01 {

static Logger logger = LogManager.getLogger(RagAttempt01.class);

public static void main(String[] args) {
    try {

	logger.info("Started");

	logger.info("Completed");

    } catch (Exception e) {
	logger.error(e.getMessage(), e);
	System.exit(1);
    }
    System.exit(0);
}

}
