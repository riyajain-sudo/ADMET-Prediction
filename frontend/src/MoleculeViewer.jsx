// Renders a molecule from plain coordinates/bonds (computed by the backend's
// ml/depict.py, via RDKit) using ordinary SVG -- no chemistry library needed
// here at all. Atoms can be highlighted by a signed "contribution" score
// (from SHAP): blue means that atom pushed the selected prediction up, red
// means it pushed the prediction down, and opacity encodes how strongly.

const PADDING = 1.2; // molecule-space units of empty space around the structure
const PX_PER_UNIT = 55; // pixels per RDKit coordinate unit
const BOND_OFFSET = 3.5; // px between parallel lines of a double/triple bond

function toLayout(atoms) {
  const xs = atoms.map((a) => a.x);
  const ys = atoms.map((a) => a.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const width = (maxX - minX + 2 * PADDING) * PX_PER_UNIT;
  const height = (maxY - minY + 2 * PADDING) * PX_PER_UNIT;

  const toSvg = (x, y) => ({
    x: (x - minX + PADDING) * PX_PER_UNIT,
    y: (maxY - y + PADDING) * PX_PER_UNIT, // flip: chemistry y-up -> SVG y-down
  });

  return { width, height, toSvg };
}

function BondLine({ begin, end, order }) {
  const dx = end.x - begin.x;
  const dy = end.y - begin.y;
  const length = Math.hypot(dx, dy) || 1;
  const px = (-dy / length) * BOND_OFFSET; // perpendicular unit vector, scaled
  const py = (dx / length) * BOND_OFFSET;

  const lineCount = order >= 3 ? 3 : order >= 2 ? 2 : 1;
  const offsets =
    lineCount === 1 ? [0] : lineCount === 2 ? [-0.5, 0.5] : [-1, 0, 1];

  return (
    <>
      {offsets.map((k, i) => (
        <line
          key={i}
          x1={begin.x + px * k}
          y1={begin.y + py * k}
          x2={end.x + px * k}
          y2={end.y + py * k}
          stroke="var(--ink-secondary)"
          strokeWidth={2}
        />
      ))}
    </>
  );
}

export default function MoleculeViewer({ structure, contributions, unmappedContribution }) {
  const { atoms, bonds } = structure;
  const { width, height, toSvg } = toLayout(atoms);
  const positions = atoms.map((a) => toSvg(a.x, a.y));

  const maxAbsContribution = contributions
    ? Math.max(1e-9, ...Object.values(contributions).map(Math.abs))
    : 0;

  return (
    <div className="molecule-viewer">
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={Math.min(height, 320)}>
        {contributions &&
          atoms.map((atom, i) => {
            const score = contributions[atom.index];
            if (!score) return null;
            const opacity = 0.18 + 0.75 * (Math.abs(score) / maxAbsContribution);
            return (
              <circle
                key={`highlight-${atom.index}`}
                cx={positions[i].x}
                cy={positions[i].y}
                r={16}
                fill={score > 0 ? "var(--up)" : "var(--down)"}
                opacity={opacity}
              />
            );
          })}

        {bonds.map((bond, i) => (
          <BondLine
            key={i}
            begin={positions[bond.begin_atom]}
            end={positions[bond.end_atom]}
            order={bond.order}
          />
        ))}

        {atoms.map((atom, i) => (
          <g key={atom.index}>
            <circle cx={positions[i].x} cy={positions[i].y} r={3} fill="var(--ink-secondary)" />
            {atom.element !== "C" && (
              <text
                x={positions[i].x}
                y={positions[i].y}
                dy="0.35em"
                textAnchor="middle"
                fontSize={13}
                fontWeight={600}
                fill="var(--ink-primary)"
                stroke="var(--surface)"
                strokeWidth={3}
                paintOrder="stroke"
              >
                {atom.element}
              </text>
            )}
          </g>
        ))}
      </svg>

      {contributions && (
        <div className="molecule-legend">
          <span><i className="swatch swatch-up" /> pushes prediction up</span>
          <span><i className="swatch swatch-down" /> pushes prediction down</span>
          <span>Approximate atom allocation; unmapped SHAP residual: {Number(unmappedContribution ?? 0).toFixed(3)}</span>
        </div>
      )}
    </div>
  );
}
