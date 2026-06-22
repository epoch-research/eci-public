// Node loader: stub .yml/.yaml imports so the website's eciSubsetMath.ts can be
// imported outside Vite. The fit functions (fitModelECI &c.) don't use the YAML.
export async function load(url, context, next) {
  if (url.endsWith('.yml') || url.endsWith('.yaml'))
    return { format: 'module', source: 'export default {}', shortCircuit: true };
  return next(url, context);
}
