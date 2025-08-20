
⸻

MRR Update Process

1. Run MRR Report in Cognos (by group code)
	•	Add “Month” and “Year” columns from Order Promised Date.
	•	If Order Promised Date is missing, use Need By Date.
	•	Add a “Promise Dates” flag column (Boolean = Yes/No) to indicate whether an Order Promised Date exists.

2. Run SCM Requisition Report to add Requisition Status
	•	Navigate: SCM Data Model Folder → SCM Reporting → SCM Requisition Report.
	•	Use Project Code (Column E), set as task, beginning at the most recent received date of material.
	•	Match MCPR Order Number (Column J) to Req # (Column B).
	•	Exclude rejected and returned material.

3. Pull open order report from SCM portal to add VPP Billing
	•	Match MCPR Order Number to Requisition Number.
	•	Populate data from VPP Billing Rate.

⸻

Output: Updated MRR with added columns 