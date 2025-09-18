import * as fs from 'fs/promises';
import * as path from 'path';

abstract class ImageGenerator {
    abstract generateImageFromPrompt(senderUuid: string, messageId: string, promptString: string, imageWidthInPixels: number, imageHeightInPixels: number): Promise<string>;
}

class SpectacleImageGenerator extends ImageGenerator {
    private directoryPath: string = "../image-server";

    async generateImageFromPrompt(senderUuid: string, messageId: string, promptString: string, imageWidthInPixels: number, imageHeightInPixels: number): Promise<string> {
        const baseName = `${senderUuid}-${messageId}-${imageWidthInPixels}x${imageHeightInPixels}`;
        const promptFile = path.join(this.directoryPath, `${baseName}.txt`);
        const imageFile = path.join(this.directoryPath, `${baseName}.png`);
        const errorFile = path.join(this.directoryPath, `${baseName}-error.txt`);

        // Write the prompt file
        try {
            await fs.writeFile(promptFile, promptString);
        } catch (error) {
            return `Failed to write prompt file: ${error instanceof Error ? error.message : 'Unknown error'}`;
        }

        // Wait for either the image or error file
        const maxAttempts = 300; // 5 minutes with 1s intervals
        let attempts = 0;
        while (attempts < maxAttempts) {
            try {
                await fs.access(imageFile);
                return imageFile; // Return the full path to the image file
            } catch (error) {
                // File doesn't exist yet, continue polling
            }

            try {
                await fs.access(errorFile);
                try {
                    const errorContent = await fs.readFile(errorFile, 'utf-8');
                    return `Image generation failed: ${errorContent}`;
                } catch (readError) {
                    return `Failed to read error file: ${readError instanceof Error ? readError.message : 'Unknown error'}`;
                }
            } catch (error) {
                // Error file doesn't exist yet, continue polling
            }

            await this.sleep(1000);
            attempts++;
        }

        return "Timeout: Image generation took too long.";
    }

    private sleep(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
