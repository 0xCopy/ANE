## Distributed 3way merge — D3M

D3M is the endorsed integration technique for distributed branch work. 


 1. git fetch

 2. git add . 

 3. git commit wip (or wip - plus relevant commit details is  fine here)

 4. git 3-way merge all unmerged branches n relation tro origin master in ascending order by time; UNBIASED.  UNre not 3 way merging with tip, you are sequentially merge the 3 oldest unmerged, in a loop,  BIASED means you do not keep existing, or keep new, you merge all, and retain all conflicts. 

 5. at the end you identify the intent of each conflict or blocks of conflicts for the intent and the fix.

 6. fix compilation and tests.  tests older than 30 days are no longer considered current tests.  better to delete them than fix them.  

 7. git push to origin master.  

 8. do not delete remote or local branches after merge or push,

 9. read the atmosphere during this timeline review opportunity and improve the projkect.  if a branch was abandoned or carries outdated momentum make a judgement call to improve the outcome and avoid regression and tasktree bloat.  identify good features and bad features in contentious snarls, the end-=state review is precisely to see all of the conflicts in one scene.



