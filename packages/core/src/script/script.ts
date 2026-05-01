export async function loadExternalScript(
  documentRef: Document | null,
  scriptUrl: string,
): Promise<boolean> {
  if (
    !documentRef ||
    !documentRef.createElement ||
    !documentRef.head ||
    typeof scriptUrl !== 'string' ||
    !scriptUrl.trim()
  ) {
    throw new Error('script loading is unavailable');
  }

  return new Promise<boolean>((resolve, reject) => {
    try {
      const script = documentRef.createElement('script');
      script.src = scriptUrl;
      script.onload = () => resolve(true);
      script.onerror = () => reject(new Error(`Failed to load script from ${scriptUrl}`));
      documentRef.head.appendChild(script);
    } catch (error) {
      reject(error);
    }
  });
}
