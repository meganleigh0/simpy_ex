MRR Labeling Process

Run MRR Report in Cognos

Go to: SCM Data Model Folder > SCM Reporting > MRR Report

Run the MRR by group code.

Run SCM Requisition Report

Go to: SCM Data Model Folder > SCM Reporting > SCM Requisition Report

Begin at the latest received date of material.

Exclude rejected and returned material (Column E = Project Code).

Match Orders

Match MCPR Order Number (Column J) with Requisition # (Column B).

Keep only true planned replenishments and commitments.

Pull Open Order Report

From the SCM portal, run the Open Order Report.

Match MCPR Order Number (Column J).

Identify vendor progress payments based on MCPR Order Numbers.

Result

Output = Labeled MRR (with validated requisitions, commitments, and progress payments).
