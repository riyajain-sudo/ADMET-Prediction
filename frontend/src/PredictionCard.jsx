// A clickable summary card for one predicted property. Clicking it tells
// the parent to highlight the molecule using this property's SHAP scores.
// A plain div (not <button>) because the Tox21 card nests a <select> inside,
// which isn't valid inside a real button element.
export default function PredictionCard({ title, valueText, description, active, onClick, children }) {
  return (
    <div
      role="button"
      tabIndex={0}
      className={`prediction-card${active ? " active" : ""}`}
      onClick={onClick}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onClick()}
    >
      <div className="prediction-card-title">{title}</div>
      <div className="prediction-card-value">{valueText}</div>
      <div className="prediction-card-description">{description}</div>
      {children}
    </div>
  );
}
