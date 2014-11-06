<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>

	<div class="row">
		<?php echo $form->labelEx($model,'Fecha'); ?>
<?php echo $form->textField($model,'Fecha'); ?>
<?php echo $form->error($model,'Fecha'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'Eff_Shift'); ?>
<?php echo $form->textField($model,'Eff_Shift'); ?>
<?php echo $form->error($model,'Eff_Shift'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'Count_Fill_Batch'); ?>
<?php echo $form->textField($model,'Count_Fill_Batch'); ?>
<?php echo $form->error($model,'Count_Fill_Batch'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'Count_Pall_Batch'); ?>
<?php echo $form->textField($model,'Count_Pall_Batch'); ?>
<?php echo $form->error($model,'Count_Pall_Batch'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'Count_Fill_Shift'); ?>
<?php echo $form->textField($model,'Count_Fill_Shift'); ?>
<?php echo $form->error($model,'Count_Fill_Shift'); ?>
	</div>


