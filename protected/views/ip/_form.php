<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>

	<div class="row">
		<?php echo $form->labelEx($model,'Fecha'); ?>
<?php echo $form->textField($model,'Fecha'); ?>
<?php echo $form->error($model,'Fecha'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'TAG'); ?>
<?php echo $form->textField($model,'TAG',array('size'=>50,'maxlength'=>50)); ?>
<?php echo $form->error($model,'TAG'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'IP'); ?>
<?php echo $form->textField($model,'IP'); ?>
<?php echo $form->error($model,'IP'); ?>
	</div>


