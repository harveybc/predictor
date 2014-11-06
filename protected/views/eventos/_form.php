<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>

	<div class="row">
		<?php echo $form->labelEx($model,'usuario'); ?>
<?php echo $form->textField($model,'usuario',array('size'=>31,'maxlength'=>31)); ?>
<?php echo $form->error($model,'usuario'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'modulo'); ?>
<?php echo $form->textField($model,'modulo',array('size'=>32,'maxlength'=>32)); ?>
<?php echo $form->error($model,'modulo'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'operacion'); ?>
<?php echo $form->textField($model,'operacion',array('size'=>32,'maxlength'=>32)); ?>
<?php echo $form->error($model,'operacion'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'ip'); ?>
<?php echo $form->textField($model,'ip',array('size'=>12,'maxlength'=>12)); ?>
<?php echo $form->error($model,'ip'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'descripcion'); ?>
<?php echo $form->textField($model,'descripcion',array('size'=>60,'maxlength'=>255)); ?>
<?php echo $form->error($model,'descripcion'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'fecha'); ?>
<?php echo $form->textField($model,'fecha'); ?>
<?php echo $form->error($model,'fecha'); ?>
	</div>


