import * as fs from 'fs/promises';
import * as path from 'path';

/**
 * Represents the result of an image generation operation.
 * @typedef {Object} ImageGenerationResult
 * @property {'success' | 'error'} status - Indicates whether the image generation was successful or failed.
 * @property {string} [imagePath] - The full file path to the generated image file, present only if status is 'success'.
 * @property {string} [message] - The error message describing the failure, present only if status is 'error'.
 */
type ImageGenerationResult =
  | { status: 'success'; imagePath: string }
  | { status: 'error'; message: string };

/**
 * Abstract base class for image generation implementations.
 */
abstract class ImageGenerator {
    /**
     * Generates an image based on a provided prompt and saves it to a file.
     * @abstract
     * @param {string} senderUuid - The UUID of the user requesting the image.
     * @param {string} messageId - The unique ID of the message associated with the request.
     * @param {string} promptString - The text prompt describing the desired image.
     * @param {number} imageWidthInPixels - The width of the image in pixels.
     * @param {number} imageHeightInPixels - The height of the image in pixels.
     * @returns {Promise<ImageGenerationResult>} A promise that resolves to an object indicating success with the image file path or an error with a message.
     */
    abstract generateImageFromPrompt(
        senderUuid: string,
        messageId: string,
        promptString: string,
        imageWidthInPixels: number,
        imageHeightInPixels: number
    ): Promise<ImageGenerationResult>;
}

/**
 * Implementation of ImageGenerator that uses Qwen-Image to generate images.
 * Interacts with Spectacle server by writing prompt files and polling for results.
 */
class QwenImageGenerator extends ImageGenerator {
    private directoryPath: string = "../image-server";

    /**
     * Creates an instance of QwenImageGenerator.
     * @param {string} directoryPath - The directory where prompt and image files are stored.
     */
    constructor(directoryPath: string) {
        super();
        this.directoryPath = directoryPath;
    }

    /**
     * Generates an image by writing a prompt file and polling for the resulting image or error file.
     * @param {string} senderUuid - The UUID of the user requesting the image.
     * @param {string} messageId - The unique ID of the message associated with the request.
     * @param {string} promptString - The text prompt describing the desired image.
     * @param {number} imageWidthInPixels - The width of the image in pixels.
     * @param {number} imageHeightInPixels - The height of the image in pixels.
     * @returns {Promise<ImageGenerationResult>} A promise that resolves to an object indicating success with the image file path or an error with a message.
     */
    async generateImageFromPrompt(
        senderUuid: string,
        messageId: string,
        promptString: string,
        imageWidthInPixels: number,
        imageHeightInPixels: number
    ): Promise<ImageGenerationResult> {
        const baseName = `${senderUuid}-${messageId}-${imageWidthInPixels}x${imageHeightInPixels}`;
        const promptFile = path.join(this.directoryPath, `${baseName}.txt`);
        const imageFile = path.join(this.directoryPath, `${baseName}.png`);
        const errorFile = path.join(this.directoryPath, `${baseName}-error.txt`);

        // Write the prompt file
        try {
            await fs.writeFile(promptFile, promptString);
        } catch (error) {
            return {
                status: 'error',
                message: `Failed to write prompt file: ${error instanceof Error ? error.message : 'Unknown error'}`
            };
        }

        // Wait for either the image or error file
        const maxAttempts = 300; // 5 minutes with 1s intervals
        let attempts = 0;
        while (attempts < maxAttempts) {
            try {
                await fs.access(imageFile);
                return { status: 'success', imagePath: imageFile }; // Return success with image file path
            } catch (error) {
                // File doesn't exist yet, continue polling
            }

            try {
                await fs.access(errorFile);
                try {
                    const errorContent = await fs.readFile(errorFile, 'utf-8');
                    return { status: 'error', message: `Image generation failed: ${errorContent}` };
                } catch (readError) {
                    return {
                        status: 'error',
                        message: `Failed to read error file: ${readError instanceof Error ? readError.message : 'Unknown error'}`
                    };
                }
            } catch (error) {
                // Error file doesn't exist yet, continue polling
            }

            await this.sleep(1000);
            attempts++;
        }

        return { status: 'error', message: 'Timeout: Image generation took too long.' };
    }

    /**
     * Pauses execution for the specified number of milliseconds.
     * @param {number} ms - The number of milliseconds to sleep.
     * @returns {Promise<void>} A promise that resolves after the specified delay.
     */
    private sleep(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
