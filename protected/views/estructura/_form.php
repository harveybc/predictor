
<p class="note">Campos con <span class="required">*</span> son necesarios.</p>


<div class="forms50cb">
    <div>
        <div>
            <div>
                
                <div class="row">
                    <?php echo $form->labelEx($model, 'Codigo'); ?>
                    <?php echo $form->textField($model, 'Codigo'); ?>
                    <?php echo $form->error($model, 'Codigo'); ?>
                </div>
            </div>
        </div>

        <div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'Equipo'); ?>
                    <?php echo $form->textField($model, 'Equipo', array('size' => 50, 'maxlength' => 50,)); ?>
                    <?php echo $form->error($model, 'Equipo'); ?>
                </div>
            </div>
        </div>

        <div>
            <div>
                <div class="row">
                    <?php /*echo $form->labelEx($model, 'plan_mant_ultrasonido'); ?>
                    <?php echo $form->textField($model, 'plan_mant_ultrasonido'); ?>
                    <?php echo $form->error($model, 'plan_mant_ultrasonido'); */?>
                </div>
            </div>
        </div>

    </div>

</div>

<?php echo $form->errorSummary($model); ?>










