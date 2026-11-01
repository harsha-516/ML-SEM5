// Taken from infrared satellite enhancement: colder, stronger cloud tops step toward magenta.
export const RAMP = ['#3B8FD0', '#2FC4C0', '#E8DC4E', '#F58A2C', '#E5264F', '#D24BE6']

export const STOPS = [0.01, 0.03, 0.08, 0.16, 0.3, 0.5]

export function rampExpression(prop: string): unknown[] {
  const e: unknown[] = ['interpolate', ['linear'], ['get', prop]]
  STOPS.forEach((s, i) => e.push(s, RAMP[i]))
  return e
}
